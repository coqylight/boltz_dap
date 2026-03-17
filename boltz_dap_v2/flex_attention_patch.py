"""
FlexAttention integration for Boltz2 triangle attention.

Replaces the naive _attention (which materializes full QK^T + bias matrix)
with PyTorch's FlexAttention that fuses bias into the kernel.

Benefits:
  - No full attention matrix materialized (memory savings)
  - Deterministic (same kernel on every GPU)
  - Handles arbitrary bias via score_mod

Usage:
    from flex_attention_patch import patch_triangle_attention
    patch_triangle_attention(model)  # call once after model creation
"""

import math
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

# Use flex_attention directly. torch.compile(flex_attention) at import time triggers
# PyTorch inductor's TritonTemplate registration and raises "duplicate template name"
# (torch/_inductor/select_algorithm.py assert name not in self.all_templates).
# So we do not compile here; use flex_attention as-is for correctness.
_flex_attention_fn = flex_attention


def _flex_attention_forward(
    self,
    q_x: Tensor,
    kv_x: Tensor,
    tri_bias: Tensor,
    mask_bias: Tensor,
    mask: Tensor,
    use_kernels: bool = False,
) -> Tensor:
    """Drop-in replacement for Attention.forward using FlexAttention.

    Pre-combines biases into a single [B_eff, H, Q, K] tensor so score_mod
    does a simple 4-index lookup: bias[b, h, q_idx, kv_idx].

    Yes, this materializes the combined bias — but it replaces the
    even-larger QK^T attention matrix that the naive path materializes.
    Net effect: same or less memory, fused softmax.
    """
    # Get q, k, v WITHOUT scaling (FlexAttention handles scale internally)
    q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=False)

    ndim = q.dim()
    c_h = q.shape[-1]
    scale = 1.0 / math.sqrt(c_h)

    if ndim == 4:
        # ---- Chunked path ----
        # q,k,v: [chunk_size, H, J, c_h]
        # tri_bias: [chunk_size, H, J, J]
        # mask_bias: [chunk_size, 1, 1, J]

        # Pre-combine so score_mod is a single index lookup
        # tri_bias already [cs, H, J, J], mask_bias broadcasts
        bias = tri_bias + mask_bias  # [cs, H, J, J]

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        o = _flex_attention_fn(q, k, v, score_mod=score_mod, scale=scale)
        o = o.transpose(-2, -3)  # [cs, J, H, c_h]

    else:
        # ---- Non-chunked path (5D) ----
        B, I, H, J, _ = q.shape
        # DAP chunked path: tri_bias can be full [B, 1, H, N, N] while q is [B, chunk_I, H, J, _].
        # Slice tri_bias to this rank's rows so FlexAttention can be used.
        if tri_bias.shape[3] != I or tri_bias.shape[4] != J:
            N = tri_bias.shape[3]
            dap_size = 1
            try:
                from boltz_distributed.core import get_dap_size
                dap_size = get_dap_size()
            except Exception:
                pass
            # DAP chunked input (I < N): FlexAttention's dense path materializes Q@K^T
            # [B*I, H, J, J] = e.g. 779*4*3114*3114*4 bytes ~112GB → OOM. Use original.
            if dap_size > 1:
                if hasattr(self, "_flex_original_forward"):
                    return self._flex_original_forward(
                        q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=use_kernels
                    )
                raise RuntimeError(
                    "FlexAttention 5D path: DAP chunked (dap_size>1) requires _flex_original_forward fallback."
                )
            # Single GPU: shape mismatch but not DAP → fallback
            if hasattr(self, "_flex_original_forward"):
                return self._flex_original_forward(
                    q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=use_kernels
                )
            raise RuntimeError(
                "FlexAttention 5D path: tri_bias shape [..., %s, %s] does not match q I=%s J=%s."
                % (tri_bias.shape[3], tri_bias.shape[4], I, J)
            )

        q = q.reshape(B * I, H, J, c_h)
        k = k.reshape(B * I, H, J, c_h)
        v = v.reshape(B * I, H, J, c_h)

        # tri_bias: [B, 1, H, I, J] -> [B, I, H, I, J] -> [B*I, H, I, J]
        # I = J = N, so this is [B*I, H, J, J]
        _tri = tri_bias.expand(B, I, H, I, J).reshape(B * I, H, I, J)
        # mask_bias: [B, I, 1, 1, J] -> [B*I, 1, 1, J]
        _mask = mask_bias.reshape(B * I, 1, 1, J)

        bias = _tri + _mask  # [B*I, H, J, J]
        del _tri, _mask

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]

        o = _flex_attention_fn(q, k, v, score_mod=score_mod, scale=scale)
        o = o.reshape(B, I, H, J, c_h)
        o = o.transpose(-2, -3)  # [B, I, J, H, c_h]

    o = self._wrap_up(o, q_x)
    return o


def patch_triangle_attention(model):
    """Monkey-patch all TriangleAttention modules to use FlexAttention."""
    import types
    from boltz.model.layers.triangular_attention.attention import TriangleAttention

    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, TriangleAttention):
            # Save original forward for fallback when DAP chunked (tri_bias shape != q chunk shape)
            module.mha._flex_original_forward = module.mha.forward
            module.mha.forward = types.MethodType(
                _flex_attention_forward, module.mha
            )
            patched += 1

    return patched
