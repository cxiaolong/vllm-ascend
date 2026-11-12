# SPDX-License-Identifier: Apache-2.0
"""Preserve real adapter slots during idle DP participation in Model Runner V2."""

import inspect
from contextlib import contextmanager
from functools import wraps


def install_idle_dp_handling(runner_cls):
    """Install once per runner class; each runner owns its profiling state.

    This is installed by the worker patch after NPU runner imports are ready.
    Bind upstream signatures instead of assuming profiling flags' positions.
    """
    if runner_cls.__dict__.get("_kimi_idle_lora_installed", False):
        return
    required = ("_dummy_run", "maybe_dummy_run_with_lora", "_set_active_loras")
    if any(not callable(getattr(runner_cls, name, None)) for name in required):
        raise RuntimeError("Installed vLLM runner lacks the Kimi idle LoRA lifecycle hooks")
    original_dummy = runner_cls._dummy_run
    original_lora = runner_cls.maybe_dummy_run_with_lora
    signature = inspect.signature(original_dummy)

    @wraps(original_dummy)
    def dummy(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        previous = getattr(self, "_kimi_profile_dummy", False)
        self._kimi_profile_dummy = bound.arguments.get("is_profile", False)
        try:
            return original_dummy(self, *args, **kwargs)
        finally:
            self._kimi_profile_dummy = previous

    @contextmanager
    def lora_context(self, lora_config, num_scheduled_tokens, num_sampled_tokens=None, remove_lora=True, **kwargs):
        manager = getattr(getattr(self, "lora_manager", None), "_adapter_manager", None)
        is_kimi_dp = hasattr(manager, "_kimi_banks") and self.vllm_config.parallel_config.data_parallel_size > 1
        if not is_kimi_dp or getattr(self, "_kimi_profile_dummy", False):
            with original_lora(self, lora_config, num_scheduled_tokens, num_sampled_tokens, remove_lora, **kwargs):
                yield
            return
        samples = len(num_scheduled_tokens) if num_sampled_tokens is None else int(num_sampled_tokens.sum())
        self._set_active_loras((0,) * samples, (0,) * int(num_scheduled_tokens.sum()), set())
        yield

    runner_cls._dummy_run = dummy
    runner_cls.maybe_dummy_run_with_lora = lora_context
    runner_cls._kimi_idle_lora_installed = True
