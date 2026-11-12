# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native eager LoRA linear path for Kimi acceptance on images without SGMV."""

import torch

from .token_slots import slots_for_tokens


def native_lora_linear(self, y, x, lora_a_stacked, lora_b_stacked, scale, output_slices, *, buffer=None, **kwargs):
    if self.no_lora:
        return y
    if buffer is not None or not kwargs.get("add_inputs", True):
        raise ValueError("Native Kimi LoRA supports additive, non-sharded linear only")
    flat_x = x.reshape(-1, x.shape[-1])
    flat_y = y.reshape(-1, y.shape[-1])
    slots = slots_for_tokens(self, flat_x.shape[0])
    if slots.numel() != flat_x.shape[0] or flat_y.shape[0] != flat_x.shape[0]:
        raise ValueError("Native LoRA token metadata does not match linear inputs")
    if len(lora_a_stacked) != len(lora_b_stacked) or len(output_slices) != len(lora_a_stacked):
        raise ValueError("Native LoRA packed slices mismatch")
    num_slots = lora_a_stacked[0].shape[0]
    if bool(((slots < -1) | (slots >= num_slots)).any()):
        raise ValueError("Native LoRA adapter index is outside allocated slots")
    offset = 0
    for a_bank, b_bank, width in zip(lora_a_stacked, lora_b_stacked, output_slices):
        if a_bank.shape[0] != num_slots or b_bank.shape[0] != num_slots:
            raise ValueError("Native LoRA slot counts mismatch")
        target = flat_y[:, offset : offset + width]
        if target.shape[1] != width:
            raise ValueError("Native LoRA output width mismatch")
        for slot in range(num_slots):
            a = a_bank[slot].reshape(a_bank.shape[-2:]).float()
            b = b_bank[slot].reshape(b_bank.shape[-2:])[:width].float()
            if a.shape[1] != flat_x.shape[1] or b.shape != (width, a.shape[0]):
                raise ValueError("Native LoRA matrix dimensions mismatch")
            delta = (flat_x.float() @ a.T) @ b.T
            updated = torch.where((slots == slot).unsqueeze(1), target.float() + delta * scale, target.float())
            target.copy_(updated.to(target.dtype))
        offset += width
    if offset != flat_y.shape[1]:
        raise ValueError("Native LoRA slices do not cover the linear output")
    return y
