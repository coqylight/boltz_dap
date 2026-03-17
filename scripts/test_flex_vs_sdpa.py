#!/usr/bin/env python3
"""Test FlexAttention vs SDPA: correctness, memory, and timing comparison.

Run on a GPU node with the boltz_uv environment:
    source /project/engvimmune/gleeai/envs/boltz_uv/bin/activate
    python /project/engvimmune/gleeai/boltz_dap/test_flex_vs_sdpa.py
"""

import math
import time
import torch
import torch.nn.functional as F

# ── Config ──
B, H, N, ch = 1, 4, 2595, 32   # matches real pentamer shapes
DTYPE = torch.bfloat16
DEVICE = "cuda:0"
N_WARMUP = 2
N_RUNS = 5
ATOL = 1e-3

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SM: {torch.cuda.get_device_capability(0)}")
print(f"\nTest config: B={B}, H={H}, N={N}, ch={ch}, dtype={DTYPE}")
print(f"Bias shape: [{B}, {H}, {N}, {N}]")
print(f"Q/K/V shape: [{B}, {H}, {N}, {ch}]")
print()

# ── Create inputs ──
torch.manual_seed(42)
q = torch.randn(B, H, N, ch, dtype=DTYPE, device=DEVICE)
k = torch.randn(B, H, N, ch, dtype=DTYPE, device=DEVICE)
v = torch.randn(B, H, N, ch, dtype=DTYPE, device=DEVICE)
# Note: in real DAP code, bias is computed in float32 then cast to x.dtype
bias = (torch.randn(B, H, N, N, dtype=torch.float32, device=DEVICE) * 0.1).to(DTYPE)
scale = 1.0 / math.sqrt(ch)

print(f"Input memory: {(q.nelement() + k.nelement() + v.nelement()) * q.element_size() / 1e6:.1f} MB (QKV)")
print(f"Bias memory:  {bias.nelement() * bias.element_size() / 1e6:.1f} MB")
print()


# ── Helpers ──
def measure_memory():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated(0) // (1024 * 1024)


def measure_peak():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(0) // (1024 * 1024)


# ══════════════════════════════════════════════════════════════════
# 1. SDPA (EFFICIENT_ATTENTION) — current baseline
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("SDPA (EFFICIENT_ATTENTION)")
print("=" * 60)

from torch.nn.attention import sdpa_kernel, SDPBackend

# Warmup
for _ in range(N_WARMUP):
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
    torch.cuda.synchronize()

# Measure memory
torch.cuda.reset_peak_memory_stats(0)
alloc_before_sdpa = measure_memory()
with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
    out_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
alloc_after_sdpa = measure_memory()
peak_sdpa = measure_peak()
transient_sdpa = peak_sdpa - alloc_before_sdpa

print(f"  alloc before: {alloc_before_sdpa} MB")
print(f"  alloc after:  {alloc_after_sdpa} MB")
print(f"  peak:         {peak_sdpa} MB")
print(f"  transient:    {transient_sdpa} MB")

# Measure timing
times_sdpa = []
for _ in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        _ = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)
    torch.cuda.synchronize()
    times_sdpa.append((time.perf_counter() - t0) * 1000)

avg_sdpa = sum(times_sdpa) / len(times_sdpa)
print(f"  timing ({N_RUNS} runs): {avg_sdpa:.2f} ms avg")
print()

# ══════════════════════════════════════════════════════════════════
# 2. FlexAttention — new approach
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("FlexAttention (score_mod)")
print("=" * 60)

from torch.nn.attention.flex_attention import flex_attention

# Compile
print("  Compiling FlexAttention kernel (one-time)...", end="", flush=True)
t_compile_start = time.perf_counter()
compiled_flex = torch.compile(flex_attention, dynamic=False)

# First call triggers compilation
def make_score_mod(bias_tensor):
    def score_mod(score, batch, head, q_idx, k_idx):
        return score + bias_tensor[batch, head, q_idx, k_idx]
    return score_mod

score_mod_fn = make_score_mod(bias)
_ = compiled_flex(q, k, v, score_mod=score_mod_fn, scale=scale)
torch.cuda.synchronize()
compile_time = time.perf_counter() - t_compile_start
print(f" done ({compile_time:.1f}s)")

# Warmup (post-compile)
for _ in range(N_WARMUP):
    score_mod_fn = make_score_mod(bias)
    _ = compiled_flex(q, k, v, score_mod=score_mod_fn, scale=scale)
    torch.cuda.synchronize()

# Measure memory
torch.cuda.reset_peak_memory_stats(0)
alloc_before_flex = measure_memory()
score_mod_fn = make_score_mod(bias)
out_flex = compiled_flex(q, k, v, score_mod=score_mod_fn, scale=scale)
alloc_after_flex = measure_memory()
peak_flex = measure_peak()
transient_flex = peak_flex - alloc_before_flex

print(f"  alloc before: {alloc_before_flex} MB")
print(f"  alloc after:  {alloc_after_flex} MB")
print(f"  peak:         {peak_flex} MB")
print(f"  transient:    {transient_flex} MB")

# Measure timing
times_flex = []
for _ in range(N_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    score_mod_fn = make_score_mod(bias)
    _ = compiled_flex(q, k, v, score_mod=score_mod_fn, scale=scale)
    torch.cuda.synchronize()
    times_flex.append((time.perf_counter() - t0) * 1000)

avg_flex = sum(times_flex) / len(times_flex)
print(f"  timing ({N_RUNS} runs): {avg_flex:.2f} ms avg (warm, post-compile)")
print()

# ══════════════════════════════════════════════════════════════════
# 3. Correctness comparison
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("CORRECTNESS COMPARISON")
print("=" * 60)

diff = (out_sdpa.float() - out_flex.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
passed = max_diff < ATOL

print(f"  Max abs diff:  {max_diff:.2e} (tolerance: {ATOL})")
print(f"  Mean abs diff: {mean_diff:.2e}")
print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
print()

# ══════════════════════════════════════════════════════════════════
# 4. Side-by-side summary
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("SIDE-BY-SIDE SUMMARY")
print("=" * 60)
print(f"{'':18s} | {'SDPA':>10s} | {'FlexAttn':>10s} | {'Δ':>10s}")
print(f"{'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
print(f"{'alloc before (MB)':18s} | {alloc_before_sdpa:>10d} | {alloc_before_flex:>10d} | {alloc_before_flex - alloc_before_sdpa:>+10d}")
print(f"{'alloc after (MB)':18s} | {alloc_after_sdpa:>10d} | {alloc_after_flex:>10d} | {alloc_after_flex - alloc_after_sdpa:>+10d}")
print(f"{'peak (MB)':18s} | {peak_sdpa:>10d} | {peak_flex:>10d} | {peak_flex - peak_sdpa:>+10d}")
print(f"{'transient (MB)':18s} | {transient_sdpa:>10d} | {transient_flex:>10d} | {transient_flex - transient_sdpa:>+10d}")
print(f"{'avg time (ms)':18s} | {avg_sdpa:>10.2f} | {avg_flex:>10.2f} | {avg_flex - avg_sdpa:>+10.2f}")
print(f"{'compile time (s)':18s} | {'N/A':>10s} | {compile_time:>10.1f} | {'':>10s}")
print()

if passed:
    print("✓ All checks passed. FlexAttention is a drop-in replacement for SDPA.")
else:
    print("✗ Correctness check FAILED. Do not use FlexAttention without investigation.")
