# SPDX-License-Identifier: Apache-2.0
"""Opt-in Kimi native LoRA integration, loaded lazily in each vLLM process."""


def register():
    from vllm_ascend import envs

    if not envs.VLLM_ASCEND_ENABLE_KIMI_LORA:
        return
    from .registration import install

    install()
