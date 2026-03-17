"""
FlexAttention with chunked QKV for DAP (experimental).

Same as flex_attention_patch but when dap_size > 1, instead of falling back
to original TriangleAttention, runs FlexAttention in sub-chunks over the
query dimension to avoid materializing full [I, J, J] (OOM).

Backup of the non-chunked baseline: backup/flex_attention_patch_baseline.py
"""

import math
import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

_flex_attention_fn = flex_attention

# Sub-chunk size for query dimension when DAP is active. Larger = faster, fewer kernel launches.
# PyTorch flex_attention (uncompiled) materializes ~2x score matrix; bias [B*C,H,J,J] also scales with C.
# 128 = try for speed (may OOM on 80GB); if OOM, reduce to 64 or 32.
FLEX_DAP_CHUNK = 64


def _flex_attention_forward(
    self,
    q_x: Tensor,
    kv_x: Tensor,
    tri_bias: Tensor,
    mask_bias: Tensor,
    mask: Tensor,
    use_kernels: bool = False,
) -> Tensor:
    q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=False)
    ndim = q.dim()
    c_h = q.shape[-1]
    scale = 1.0 / math.sqrt(c_h)

    if ndim == 4:
        bias = tri_bias + mask_bias
        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias[b, h, q_idx, kv_idx]
        o = _flex_attention_fn(q, k, v, score_mod=score_mod, scale=scale)
        o = o.transpose(-2, -3)
    else:
        # ---- 5D path ----
        B, I, H, J, _ = q.shape
        if tri_bias.shape[3] != I or tri_bias.shape[4] != J:
            N = tri_bias.shape[3]
            dap_size = 1
            try:
                from boltz_distributed.core import get_dap_size, get_dap_rank
                dap_size = get_dap_size()
                dap_rank = get_dap_rank()
            except Exception:
                dap_rank = 0

            if dap_size > 1 and tri_bias.shape[4] == J and N >= I:
                # Chunked FlexAttention: split I into sub-chunks to avoid full [I,J,J] allocation.
                split_size = (N + dap_size - 1) // dap_size
                row_start = dap_rank * split_size
                chunk_size = min(FLEX_DAP_CHUNK, I)
                # One-time log: if OOM alloc is same for 128 and 256, cause may be elsewhere (e.g. fixed buffer).
                if not hasattr(_flex_attention_forward, "_chunk_logged"):
                    _flex_attention_forward._chunk_logged = True
                    try:
                        from boltz_distributed.core import get_dap_rank
                        if get_dap_rank() == 0:
                            one_score_gib = (chunk_size * 4 * J * J * 4) / (1024**3)
                            print(f"[FlexAttention chunked] FLEX_DAP_CHUNK={FLEX_DAP_CHUNK}, I={I}, J={J}, first C={chunk_size}, one_score_matrix_GiB≈{one_score_gib:.2f}", flush=True)
                    except Exception:
                        pass
                out_list = []
                for start in range(0, I, chunk_size):
                    end = min(start + chunk_size, I)
                    q_c = q[:, start:end]   # [B, C, H, J, c_h]
                    k_c = k[:, start:end]
                    v_c = v[:, start:end]
                    C = q_c.shape[1]
                    # Like DAP chunk_layer: use full [N,N] bias per chunk so result matches original.
                    # tri_bias [B, 1, H, N, N] -> expand so each of C batch elements gets same full [N,N].
                    mask_c = mask_bias[:, start:end]  # [B, C, 1, 1, J]
                    q_c = q_c.reshape(B * C, H, J, c_h)
                    k_c = k_c.reshape(B * C, H, J, c_h)
                    v_c = v_c.reshape(B * C, H, J, c_h)
                    # [B*C, H, J, J]: full bias (same [N,N] per batch element, then + mask)
                    bias_c = tri_bias.expand(B, C, H, J, J).reshape(B * C, H, J, J).clone()
                    bias_c = bias_c + mask_c.reshape(B * C, 1, 1, J).expand(B * C, H, J, J)
                    def score_mod_c(score, b, h, q_idx, kv_idx):
                        return score + bias_c[b, h, q_idx, kv_idx]
                    o_c = _flex_attention_fn(q_c, k_c, v_c, score_mod=score_mod_c, scale=scale)
                    o_c = o_c.reshape(B, C, H, J, c_h).transpose(-2, -3)  # [B, C, J, H, c_h]
                    out_list.append(o_c)
                o = torch.cat(out_list, dim=1)  # [B, I, J, H, c_h]
            else:
                if hasattr(self, "_flex_original_forward"):
                    return self._flex_original_forward(
                        q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=use_kernels
                    )
                raise RuntimeError(
                    "FlexAttention 5D path: tri_bias shape [..., %s, %s] does not match q I=%s J=%s."
                    % (tri_bias.shape[3], tri_bias.shape[4], I, J)
                )
        else:
            q = q.reshape(B * I, H, J, c_h)
            k = k.reshape(B * I, H, J, c_h)
            v = v.reshape(B * I, H, J, c_h)
            _tri = tri_bias.expand(B, I, H, I, J).reshape(B * I, H, I, J)
            _mask = mask_bias.reshape(B * I, 1, 1, J)
            bias = _tri + _mask
            def score_mod(score, b, h, q_idx, kv_idx):
                return score + bias[b, h, q_idx, kv_idx]
            o = _flex_attention_fn(q, k, v, score_mod=score_mod, scale=scale)
            o = o.reshape(B, I, H, J, c_h).transpose(-2, -3)

    o = self._wrap_up(o, q_x)
    return o


def patch_triangle_attention(model):
    """Monkey-patch all TriangleAttention to use chunked FlexAttention when DAP is active."""
    import types
    from boltz.model.layers.triangular_attention.attention import TriangleAttention
    patched = 0
    for name, module in model.named_modules():
        if isinstance(module, TriangleAttention):
            module.mha._flex_original_forward = module.mha.forward
            module.mha.forward = types.MethodType(_flex_attention_forward, module.mha)
            patched += 1
    return patched
