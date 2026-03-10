#!/usr/bin/env python3
"""Test FlexAttention vs SDPA with float32 bias (matching real inference)."""

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import time

torch.manual_seed(42)
device = "cuda"

B, H, N, ch = 1, 4, 2595, 32
dtype = torch.bfloat16

q = torch.randn(B, H, N, ch, device=device, dtype=dtype)
k = torch.randn(B, H, N, ch, device=device, dtype=dtype)
v = torch.randn(B, H, N, ch, device=device, dtype=dtype)

# KEY: use float32 bias like the real inference (mask_bias + tri_bias create float32)
bias_f32 = torch.randn(B, H, N, N, device=device, dtype=torch.float32) * 0.1

scale = 1.0 / (ch ** 0.5)

print(f"q dtype: {q.dtype}, bias dtype: {bias_f32.dtype}")

# ─── SDPA with float32 bias ───
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias_f32, scale=scale)

# ─── FlexAttention with float32 bias ───
compiled_flex = torch.compile(flex_attention, dynamic=False)

def score_mod(score, batch, head, q_idx, k_idx):
    return score + bias_f32[batch, head, q_idx, k_idx]

out_flex = compiled_flex(q, k, v, score_mod=score_mod, scale=scale)

# ─── Compare ───
diff = (out_sdpa - out_flex).abs()
print(f"\nWith float32 bias:")
print(f"  Max abs diff:  {diff.max().item():.6e}")
print(f"  Mean abs diff: {diff.mean().item():.6e}")
print(f"  Result: {'✓ PASS' if diff.max().item() < 0.01 else '✗ FAIL'}")

# ─── Also test with bf16 bias (our standalone test) ───
bias_bf16 = bias_f32.to(torch.bfloat16)

with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
    out_sdpa_bf16 = F.scaled_dot_product_attention(q, k, v, attn_mask=bias_bf16, scale=scale)

def score_mod_bf16(score, batch, head, q_idx, k_idx):
    return score + bias_bf16[batch, head, q_idx, k_idx]

out_flex_bf16 = compiled_flex(q, k, v, score_mod=score_mod_bf16, scale=scale)

diff_bf16 = (out_sdpa_bf16 - out_flex_bf16).abs()
print(f"\nWith bf16 bias:")
print(f"  Max abs diff:  {diff_bf16.max().item():.6e}")
print(f"  Mean abs diff: {diff_bf16.mean().item():.6e}")
print(f"  Result: {'✓ PASS' if diff_bf16.max().item() < 0.01 else '✗ FAIL'}")

# ─── Compare SDPA f32-bias vs SDPA bf16-bias ───
diff_sdpa_dtype = (out_sdpa - out_sdpa_bf16).abs()
print(f"\nSDPA f32-bias vs SDPA bf16-bias:")
print(f"  Max abs diff:  {diff_sdpa_dtype.max().item():.6e}")
print(f"  Mean abs diff: {diff_sdpa_dtype.mean().item():.6e}")

# ─── Compare flex f32-bias vs SDPA f32-bias output shapes ───
print(f"\nOutput shapes:")
print(f"  SDPA: {out_sdpa.shape}, dtype: {out_sdpa.dtype}")
print(f"  Flex: {out_flex.shape}, dtype: {out_flex.dtype}")

# ─── Stress test: multiple sequential calls to check if closure captures stale bias ───
print(f"\nStress test: multiple calls with DIFFERENT biases...")
results = []
for i in range(5):
    new_bias = torch.randn(B, H, N, N, device=device, dtype=torch.float32) * 0.1
    
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=new_bias, scale=scale)
    
    def make_score_mod(b):
        def score_mod(score, batch, head, q_idx, k_idx):
            return score + b[batch, head, q_idx, k_idx]
        return score_mod
    
    test = compiled_flex(q, k, v, score_mod=make_score_mod(new_bias), scale=scale)
    d = (ref - test).abs().max().item()
    results.append(d)
    print(f"  Call {i}: max_diff = {d:.6e}")

print(f"\n  Max of all calls: {max(results):.6e}")
print(f"  Result: {'✓ PASS' if max(results) < 0.01 else '✗ FAIL — stale closure!'}")
