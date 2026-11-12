# SPDX-License-Identifier: Apache-2.0
"""Reject modes outside the native Kimi adapter implementation's contract."""

KIMI_MODEL_TYPES = frozenset(("kimi_k3", "kimi_k31", "kimi_linear"))


def is_kimi_config(config):
    return getattr(config.model_config.hf_config, "model_type", "") in KIMI_MODEL_TYPES


def validate_configuration(config):
    parallel = config.parallel_config
    lora = config.lora_config
    if not config.model_config.enforce_eager or config.quant_config is not None:
        raise ValueError("Kimi native LoRA requires eager, unquantized execution")
    if str(config.model_config.dtype) not in ("torch.bfloat16", "bfloat16"):
        raise ValueError("Kimi native LoRA requires BF16 base weights")
    if parallel.tensor_parallel_size != 8 or parallel.data_parallel_size not in (1, 2, 16):
        raise ValueError("Kimi native LoRA currently requires TP8 and DP1, DP2 or DP16")
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or getattr(parallel, "prefill_context_parallel_size", 1) != 1
        or config.speculative_config is not None
    ):
        raise ValueError("Kimi LoRA PP/CP/speculative modes are not covered")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("Kimi native LoRA requires prefix caching disabled")
    if config.scheduler_config.max_num_seqs != 1:
        raise ValueError("Kimi native LoRA currently requires one sequence per DP rank")
    if lora.max_loras != 1 or lora.max_lora_rank != 8 or lora.fully_sharded_loras:
        raise ValueError("Kimi native LoRA requires one slot, rank 8 and non-sharded LoRA storage")
