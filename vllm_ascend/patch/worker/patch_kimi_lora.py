# SPDX-License-Identifier: Apache-2.0
"""Attach idle-adapter protection after the NPU V2 runner is available."""

from vllm_ascend import envs

if envs.VLLM_ASCEND_ENABLE_KIMI_LORA:
    from vllm_ascend.lora.kimi import register
    from vllm_ascend.lora.kimi.idle_dp import install_idle_dp_handling
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner

    register()
    install_idle_dp_handling(NPUModelRunner)
