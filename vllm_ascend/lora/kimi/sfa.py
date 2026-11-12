# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native SFA low-rank projection banks for the opt-in Kimi integration.

The provider returns one slot per input token, with -1 for base-only.
Only native SFA is covered. Token ownership is supplied by token_slots;
context parallel, graph, and speculative decoding are rejected by registration.
"""

import types
from dataclasses import dataclass

import torch


class DSAAdapterBank:
    def __init__(self, a, b, *, heads, nope_dim, value_dim, scale):
        # a: [slots, rank, latent], b: [slots, heads*(nope+value), rank]
        if a.ndim != 3 or b.ndim != 3 or a.size(0) != b.size(0) or a.size(1) != b.size(2):
            raise ValueError("invalid LoRA bank dimensions")
        if b.size(1) != heads * (nope_dim + value_dim):
            raise ValueError("LoRA B must contain this TP rank's complete local heads")
        self.a = a
        self.b = b.reshape(b.size(0), heads, nope_dim + value_dim, a.size(1))
        self.nope_dim = nope_dim
        self.value_dim = value_dim
        self.scale = torch.full((a.size(0),), float(scale), dtype=a.dtype, device=a.device)

    def set_slot(self, slot, a, b, scale):
        """Replace a slot in place; the caller must serialize updates and forwards."""
        if slot < 0 or slot >= self.a.size(0):
            raise ValueError("adapter slot out of range")
        if a.shape != self.a[slot].shape or b.shape != self.b[slot].shape:
            raise ValueError("adapter slot tensor shape mismatch")
        self.a[slot].copy_(a)
        self.b[slot].copy_(b)
        self.scale[slot] = scale

    def clear_slot(self, slot):
        if slot < 0 or slot >= self.a.size(0):
            raise ValueError("adapter slot out of range")
        self.a[slot].zero_()
        self.b[slot].zero_()
        self.scale[slot] = 0

    def _select(self, slots, tokens):
        if slots.ndim != 1 or slots.numel() != tokens:
            raise ValueError("one adapter slot is required per input token")
        if bool(((slots < -1) | (slots >= self.a.size(0))).any()):
            raise ValueError("adapter slot out of range")
        mask = (slots >= 0).to(self.a.dtype)
        index = slots.clamp(min=0).long()
        return self.a[index], self.b[index], mask * self.scale[index]

    def k_delta(self, q_nope, slots):
        a, b, mask = self._select(slots, q_nope.size(0))
        b_k = b[:, :, : self.nope_dim, :]
        low = torch.einsum("thp,thpr->thr", q_nope, b_k)
        return torch.einsum("thr,trl->thl", low, a) * mask[:, None, None]

    def v_delta(self, x, slots):
        a, b, mask = self._select(slots, x.size(0))
        b_v = b[:, :, self.nope_dim :, :]
        low = torch.einsum("thl,trl->thr", x, a)
        return torch.einsum("thr,thvr->thv", low, b_v) * mask[:, None, None]


def _attach_projections(impl, bank, slot_provider):
    """Attach to one SFA instance; caller supplies its local token slot ordering."""
    preprocess = getattr(impl, "preprocess_type", None)
    if preprocess is not None and getattr(preprocess, "name", None) != "NATIVE":
        raise ValueError("SFA LoRA requires native preprocessing")
    original_k = impl._q_proj_and_k_up_proj
    original_v = impl._v_up_proj

    def k_forward(self, x):
        base, rope = original_k(x)
        q = self.q_proj(x)[0].view(-1, self.local_num_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        return base + bank.k_delta(q_nope, slot_provider(q_nope.size(0))), rope

    def v_forward(self, x):
        latent = x.view(-1, self.local_num_heads, self.kv_lora_rank)
        base = original_v(x)
        return base + bank.v_delta(latent, slot_provider(latent.size(0))).reshape_as(base)

    impl._q_proj_and_k_up_proj = types.MethodType(k_forward, impl)
    impl._v_up_proj = types.MethodType(v_forward, impl)


@dataclass
class Binding:
    impl: object
    adapter_key: str
    tp_rank: int = 0


def attach_sfa_adapters(manager, punica, bindings, slot_provider=None):
    if any(x is not None for x in manager.lora_index_to_id):
        raise ValueError("install before activating any adapters")
    banks = []
    for binding in bindings:
        impl = binding.impl
        preprocess = getattr(impl, "preprocess_type", None)
        if preprocess is not None and getattr(preprocess, "name", None) != "NATIVE":
            raise ValueError("only native SFA preprocessing is supported")
        heads, p, v, latent = impl.local_num_heads, impl.qk_nope_head_dim, impl.v_head_dim, impl.kv_lora_rank
        slots, rank = manager.lora_config.max_loras, manager.lora_config.max_lora_rank
        device, dtype = impl.W_UK_T.device, impl.W_UK_T.dtype
        a = torch.zeros((slots, rank, latent), dtype=dtype, device=device)
        b = torch.zeros((slots, heads * (p + v), rank), dtype=dtype, device=device)
        bank = DSAAdapterBank(a, b, heads=heads, nope_dim=p, value_dim=v, scale=0)
        banks.append((binding, bank))
    if not banks:
        raise ValueError("no SFA layers supplied")

    def provider(count):
        if slot_provider is not None:
            return slot_provider(count)
        indices = punica.token_lora_indices
        if indices.numel() < count:
            raise ValueError("LoRA metadata shorter than SFA token count")
        return indices[:count]

    # Installation mutates only the explicitly passed SFA instances.
    for binding, bank in banks:
        _attach_projections(binding.impl, bank, provider)

    old_activate = manager.activate_adapter
    old_deactivate = manager._deactivate_adapter
    old_remove_all = manager.remove_all_adapters

    def deactivate(self, adapter_id):
        try:
            slot = self.lora_index_to_id.index(adapter_id)
        except ValueError:
            slot = None
        result = old_deactivate(adapter_id)
        if slot is not None:
            for _, bank in banks:
                bank.clear_slot(slot)
        return result

    def activate(self, adapter_id):
        result = old_activate(adapter_id)
        if not result:
            return result
        try:
            slot = self.lora_index_to_id.index(adapter_id)
            adapter = self._registered_adapters[adapter_id]
            for binding, bank in banks:
                layer = adapter.get_lora(binding.adapter_key)
                if layer is None:
                    raise ValueError(f"missing mandatory DSA adapter {binding.adapter_key}")
                heads, p, v = bank.b.shape[1], bank.nope_dim, bank.value_dim
                start = binding.tp_rank * heads * (p + v)
                end = start + heads * (p + v)
                a, b = layer.lora_a, layer.lora_b
                if a.shape != bank.a[slot].shape or b.ndim != 2 or b.size(0) < end or b.size(1) != a.size(0):
                    raise ValueError(f"incompatible adapter dimensions for {binding.adapter_key}")
                bank.set_slot(slot, a, b[start:end].reshape_as(bank.b[slot]), layer.scaling)
        except Exception:
            self.deactivate_adapter(adapter_id)
            raise
        return result

    def remove_all(self):
        result = old_remove_all()
        for _, bank in banks:
            for slot in range(bank.a.size(0)):
                bank.clear_slot(slot)
        return result

    manager._deactivate_adapter = types.MethodType(deactivate, manager)
    # AdapterLRUCache captured the old bound method during manager creation.
    manager._active_adapters.deactivate_fn = manager._deactivate_adapter
    manager.activate_adapter = types.MethodType(activate, manager)
    manager.remove_all_adapters = types.MethodType(remove_all, manager)
    return banks
