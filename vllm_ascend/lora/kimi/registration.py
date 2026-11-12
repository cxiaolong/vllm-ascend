# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install native/eager Kimi LoRA wrappers in each vLLM process.

Runtime patches are opt-in, process-local and installed once. Adapter manager
instances own their banks; scoped token layouts use ContextVar. Remove these
wrappers when upstream Kimi exposes equivalent packed/SFA lifecycle hooks.
"""


def install():
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.lora.model_manager import LoRAModelManager

    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
    from vllm_ascend.models.kimi_k3.kimi_k3 import (
        AscendKimiK3ForConditionalGeneration,
        AscendKimiK3RoutedLinear,
        AscendKimiLinearForCausalLM,
        AscendKimiLinearModel,
        AscendKimiMLP,
        AscendKimiMoE,
        AscendKimiRoutedOutputTransform,
    )

    from .configuration import is_kimi_config, validate_configuration
    from .sfa import Binding, attach_sfa_adapters

    if getattr(LoRAModelManager, "_kimi_online_installed", False):
        return
    original_model_forward = AscendKimiLinearModel.forward
    original_routed_forward = AscendKimiRoutedOutputTransform.forward

    def model_forward(self, input_ids, positions, *args, **kwargs):
        from .token_slots import model_token_layout

        with model_token_layout(positions.shape[0]):
            return original_model_forward(self, input_ids, positions, *args, **kwargs)

    def routed_forward(self, hidden_states):
        if getattr(self, "_kimi_sequence_parallel", False):
            from vllm.distributed import get_tensor_model_parallel_world_size

            from .token_slots import routed_sp_layout

            with routed_sp_layout(get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()):
                return original_routed_forward(self, hidden_states)
        return original_routed_forward(self, hidden_states)

    AscendKimiLinearModel.forward = model_forward
    AscendKimiRoutedOutputTransform.forward = routed_forward
    original_dense_forward = AscendKimiMLP.forward

    def dense_forward(self, x):
        from vllm.distributed import get_tensor_model_parallel_world_size

        from .token_slots import dense_sp_layout

        if self.use_sequence_parallel:
            with dense_sp_layout(get_tensor_model_parallel_world_size()):
                return original_dense_forward(self, x)
        return original_dense_forward(self, x)

    AscendKimiMLP.forward = dense_forward
    for model_type in (AscendKimiK3ForConditionalGeneration, AscendKimiLinearForCausalLM):
        model_type.supports_lora = True
        model_type.embedding_modules = {}
        model_type.lora_skip_prefixes = []
        model_type.is_3d_moe_weight = False
        model_type.is_non_gated_moe = False
        model_type.lora_manager = None
    AscendKimiK3ForConditionalGeneration.packed_modules_mapping = AscendKimiLinearForCausalLM.packed_modules_mapping
    original_resolve = AscendSFAImpl._resolve_preprocess_type

    def resolve(self, dtype):
        config = get_current_vllm_config()
        if config.lora_config is not None and is_kimi_config(config):
            if config.quant_config is not None:
                raise ValueError("Kimi online LoRA initially requires BF16 unquantized weights")
            return PreprocessType.NATIVE
        return original_resolve(self, dtype)

    import vllm.lora.utils as lora_utils
    from vllm.lora.layers.column_parallel_linear import (
        MergedColumnParallelLinearVariableSliceWithLoRA,
        MergedColumnParallelLinearWithLoRA,
    )
    from vllm.models.kimi_k3.nvidia.kda import (
        _KimiGDNMergedColumnParallelLinear,
    )

    class KimiPackedWithLoRA(MergedColumnParallelLinearVariableSliceWithLoRA):
        @classmethod
        def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
            return (
                type(source_layer) is _KimiGDNMergedColumnParallelLinear
                and not lora_config.fully_sharded_loras
                and len(packed_modules_list) >= 3
            )

    from vllm.lora.layers.replicated_linear import ReplicatedLinearWithLoRA

    class KimiRoutedWithLoRA(ReplicatedLinearWithLoRA):
        @classmethod
        def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
            return type(source_layer) is AscendKimiK3RoutedLinear

    from vllm.model_executor.models.deepseek_v2 import DeepSeekV2FusedQkvAProjLinear

    class KimiFusedAWithLoRA(MergedColumnParallelLinearWithLoRA):
        @classmethod
        def can_replace_layer(cls, source_layer, lora_config, packed_modules_list, model_config=None):
            return (
                type(source_layer) is DeepSeekV2FusedQkvAProjLinear
                and source_layer.tp_size == 1
                and len(packed_modules_list) == 2
                and not lora_config.fully_sharded_loras
            )

    lora_utils._all_lora_classes = (KimiPackedWithLoRA, KimiRoutedWithLoRA, KimiFusedAWithLoRA) + tuple(
        lora_utils._all_lora_classes
    )

    original_init = LoRAModelManager.__init__
    original_add = LoRAModelManager._add_adapter
    original_dummy = LoRAModelManager.create_dummy_lora

    def create_dummy(self, *args, **kwargs):
        adapter = original_dummy(self, *args, **kwargs)
        # Upstream warmup creates TP-local zero weights; the absorbed SFA
        # bridge deliberately consumes global KV-B weights, as real PEFT does.
        # Only the internal dummy factory gets this shape correction.
        if hasattr(self, "_kimi_banks"):
            import torch

            for binding, bank in self._kimi_banks:
                layer = adapter.loras[binding.adapter_key]
                wrapper = self.modules[binding.adapter_key]
                layer.lora_b = torch.zeros(
                    (wrapper.base_layer.output_size, layer.lora_a.shape[0]),
                    dtype=layer.lora_b.dtype,
                    device=layer.lora_b.device,
                )
        return adapter

    def initialize(self, model, *args, **kwargs):
        is_kimi = isinstance(model, (AscendKimiK3ForConditionalGeneration, AscendKimiLinearForCausalLM))
        references = []
        if is_kimi:
            cfg = kwargs.get("vllm_config") or args[-1]
            validate_configuration(cfg)
            names = {id(module): name for name, module in reversed(list(model.named_modules()))}
            seen = set()
            for _, module in model.named_modules():
                impl = getattr(module, "impl", None)
                if isinstance(impl, AscendSFAImpl) and id(impl) not in seen:
                    seen.add(id(impl))
                    key = names.get(id(impl.kv_b_proj))
                    if key is None:
                        raise ValueError("SFA KV-B has no canonical runtime module")
                    refs = {name: value for name, value in vars(impl).items() if id(value) in names}
                    references.append((impl, key, refs))
        original_init(self, model, *args, **kwargs)
        if not is_kimi:
            return
        for module in model.modules():
            if isinstance(module, AscendKimiMoE) and module.routed_output_transform is not None:
                module.routed_output_transform._kimi_sequence_parallel = module.experts.moe_config.is_sequence_parallel
        from types import MethodType

        from .native_linear import native_lora_linear

        for punica in self.punica_wrapper_mapping.values():
            punica.add_lora_linear = MethodType(native_lora_linear, punica)
        wrappers = {id(module.base_layer): module for module in self.modules.values() if hasattr(module, "base_layer")}
        bindings = []
        for impl, key, refs in references:
            for attr, value in refs.items():
                if id(value) in wrappers:
                    setattr(impl, attr, wrappers[id(value)])
            if key not in self.modules:
                raise ValueError("Unwrapped mandatory SFA KV-B: " + key)
            bindings.append(Binding(impl, key, get_tensor_model_parallel_rank()))
        if not bindings:
            raise ValueError("Kimi LoRA requires at least one DSA layer")
        punicas = {
            id(self._get_punica_wrapper(binding.adapter_key)): self._get_punica_wrapper(binding.adapter_key)
            for binding in bindings
        }
        if len(punicas) != 1 or next(iter(punicas.values())) is None:
            raise ValueError("SFA layers require a shared language Punica mapping")
        from .token_slots import slots_for_tokens

        language_punica = next(iter(punicas.values()))
        self._kimi_banks = attach_sfa_adapters(
            self, language_punica, bindings, slot_provider=lambda count: slots_for_tokens(language_punica, count)
        )
        self._kimi_coverage = set(self.modules)
        for packed in self.packed_modules.values():
            self._kimi_coverage.update(packed)

    def add(self, adapter):
        if hasattr(self, "_kimi_coverage"):
            missing = set(adapter.loras) - self._kimi_coverage
            if missing:
                raise ValueError("Unmatched Kimi adapter modules: " + str(sorted(missing)))
        return original_add(self, adapter)

    AscendSFAImpl._resolve_preprocess_type = resolve
    LoRAModelManager.__init__ = initialize
    LoRAModelManager._add_adapter = add
    LoRAModelManager.create_dummy_lora = create_dummy
    LoRAModelManager._kimi_online_installed = True
