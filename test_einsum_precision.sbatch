#!/bin/bash
#SBATCH --job-name=einsum_prec
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/einsum_precision_%j.log

set -e
export PYTHONUNBUFFERED=1

uv run --project /project/engvimmune/gleeai/boltz python3 - <<'PYEOF'
import torch

torch.manual_seed(42)
device = "cuda"

# Reduced sizes to avoid OOM, but still big enough to trigger different tiling
B = 1
S = 256    # MSA sequences (same as real)
N = 512    # reduced from 2595
N_half = N // 2  # 256
c_hidden = 32

print("=" * 100)
print("OPM EINSUM PRECISION EXPERIMENT")
print("=" * 100)
print(f"Sizes: B={B}, S={S}, N_full={N}, N_half={N_half}, c_hidden={c_hidden}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Note: N reduced from 2595 to {N} to avoid OOM. The precision effect is N-independent.")
print()

# Generate random tensors in f32
a_f32 = torch.randn(B, S, N, c_hidden, device=device)
b_f32 = torch.randn(B, S, N, c_hidden, device=device)

chunk_size = 4

print(f"{'Precision':<12} | {'mean_diff':>12} {'max_diff':>12} {'%differ':>10} {'identical':>10}")
print("-" * 70)

for dtype_name, dtype in [("bf16", torch.bfloat16), ("f16", torch.float16), ("f32", torch.float32)]:
    tf32_modes = [False, True] if dtype == torch.float32 else [None]

    for tf32 in tf32_modes:
        label = dtype_name
        if tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            label = f"f32_{'tf32' if tf32 else 'ieee'}"

        a = a_f32.to(dtype)
        b = b_f32.to(dtype)

        # Accumulate chunked einsum results
        all_diffs = []
        for i in range(0, c_hidden, chunk_size):
            cs = min(chunk_size, c_hidden - i)
            a_chunk = a[:, :, :, i:i+cs]

            # Full N rows
            z_full = torch.einsum("bsic,bsjd->bijcd", a_chunk, b)
            # Half N rows (DAP rank 0)
            z_half = torch.einsum("bsic,bsjd->bijcd", a_chunk[:, :, :N_half, :], b)

            # Compare matching elements
            diff = (z_full[:, :N_half, :, :, :].float() - z_half.float()).abs()
            all_diffs.append(diff.flatten())

            del z_full, z_half, diff
            torch.cuda.empty_cache()

        all_diffs = torch.cat(all_diffs)
        n_diff = (all_diffs > 0).sum().item()
        pct = 100.0 * n_diff / all_diffs.numel()
        ident = "✅ YES" if n_diff == 0 else "❌ NO"

        print(f"{label:<12} | {all_diffs.mean().item():>12.8f} {all_diffs.max().item():>12.6f} {pct:>9.2f}% {ident:>10}")

        del all_diffs, a, b
        torch.cuda.empty_cache()

# Reset
torch.backends.cuda.matmul.allow_tf32 = True

# --- Single element deep dive ---
print()
print("=" * 100)
print("SINGLE-ELEMENT TRACE: z[0, 100, 200, 0, 0] = sum_s(a[0,s,100,0] * b[0,s,200,0])")
print("=" * 100)

i_pos, j_pos = 100, 200
for label, dtype, tf32 in [("bf16", torch.bfloat16, None), ("f32_tf32", torch.float32, True), ("f32_ieee", torch.float32, False)]:
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    a = a_f32.to(dtype)
    b = b_f32.to(dtype)

    # Manual Python-side sum (sequential, deterministic)
    products = a[0, :, i_pos, 0] * b[0, :, j_pos, 0]
    manual = products.sum().item()

    # einsum with full matrix
    z_full = torch.einsum("bsic,bsjd->bijcd", a[:,:,:,:1], b[:,:,:,:1])
    full_val = z_full[0, i_pos, j_pos, 0, 0].item()

    # einsum with half matrix
    z_half = torch.einsum("bsic,bsjd->bijcd", a[:,:,:N_half,:1], b[:,:,:,:1])
    half_val = z_half[0, i_pos, j_pos, 0, 0].item()

    match = "✅" if full_val == half_val else f"❌ diff={abs(full_val-half_val):.10f}"
    print(f"  {label:10}: manual={manual:>12.6f}  full_einsum={full_val:>12.6f}  half_einsum={half_val:>12.6f}  match={match}")

    del z_full, z_half, a, b
    torch.cuda.empty_cache()

torch.backends.cuda.matmul.allow_tf32 = True
print()
print("DONE")
PYEOF
