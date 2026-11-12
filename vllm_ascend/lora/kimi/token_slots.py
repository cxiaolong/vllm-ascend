# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Align LoRA token slots only against explicit attention padding metadata."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

_dense_sp_size = ContextVar("kimi_dense_sp_size", default=1)
_model_tokens = ContextVar("kimi_model_tokens", default=None)
_routed_sp = ContextVar("kimi_routed_sp", default=None)


@contextmanager
def model_token_layout(count):
    token = _model_tokens.set(count)
    try:
        yield
    finally:
        _model_tokens.reset(token)


@contextmanager
def routed_sp_layout(rank, size):
    token = _routed_sp.set((rank, size))
    try:
        yield
    finally:
        _routed_sp.reset(token)


def sharded_slots(slots, count, input_tokens, rank, size):
    """Match sp_shard's contiguous TP chunks, including rank-local padding.

    The full token count comes from the model's actual positions tensor, not
    from another DP's number of real requests or from the local row count.
    """
    if input_tokens is None or size < 1 or not 0 <= rank < size:
        raise ValueError("Missing or invalid routed SP layout")
    shard = (input_tokens + size - 1) // size
    if count != shard or slots.numel() > input_tokens:
        raise ValueError(
            f"Routed SP LoRA layout mismatch: slots={slots.numel()} rows={count} "
            f"input={input_tokens} rank={rank} size={size}"
        )
    padded = torch.cat((slots, slots.new_full((shard * size - slots.numel(),), -1)))
    return padded[rank * shard : (rank + 1) * shard]


@contextmanager
def dense_sp_layout(tp_size):
    token = _dense_sp_size.set(tp_size)
    try:
        yield
    finally:
        _dense_sp_size.reset(token)


def pad_slots(slots, count, actual_tokens, input_tokens):
    if slots.numel() >= count:
        return slots[:count]
    if slots.numel() != actual_tokens or count != input_tokens:
        raise ValueError(
            f"LoRA padding mismatch: slots={slots.numel()} rows={count} "
            f"attention_actual={actual_tokens} attention_input={input_tokens}"
        )
    return torch.cat((slots, slots.new_full((count - slots.numel(),), -1)))


def dense_sp_slots(slots, count, actual_tokens, input_tokens, tp_size):
    """Dense MLP gathers equal SP shards without trimming the final padding."""
    if tp_size < 1 or count != ((input_tokens + tp_size - 1) // tp_size) * tp_size:
        raise ValueError("Dense SP LoRA rows do not match the TP padded token layout")
    if slots.numel() != actual_tokens or actual_tokens > input_tokens:
        raise ValueError("Dense SP LoRA mapping does not match actual tokens")
    return torch.cat((slots, slots.new_full((count - slots.numel(),), -1)))


def slots_for_tokens(punica, count):
    slots = punica.token_lora_indices
    if punica.no_lora:
        return slots.new_full((count,), -1)
    if _routed_sp.get() is not None:
        rank, size = _routed_sp.get()
        return sharded_slots(slots, count, _model_tokens.get(), rank, size)
    if slots.numel() == count:
        return slots
    if slots.numel() > count and _dense_sp_size.get() == 1:
        return slots[:count]
    from vllm.forward_context import get_forward_context

    metadata = get_forward_context().attn_metadata
    candidates = metadata.values() if isinstance(metadata, dict) else [metadata]
    sfa = next(
        (
            m
            for m in candidates
            if hasattr(m, "num_input_tokens") and hasattr(m, "num_actual_tokens") and hasattr(m, "num_prefills")
        ),
        None,
    )
    if sfa is None:
        raise ValueError(f"LoRA token rows {count} exceed mapping {slots.numel()} without padding metadata")
    if _dense_sp_size.get() > 1:
        return dense_sp_slots(slots, count, sfa.num_actual_tokens, sfa.num_input_tokens, _dense_sp_size.get())
    return pad_slots(slots, count, sfa.num_actual_tokens, sfa.num_input_tokens)
