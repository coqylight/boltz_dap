#!/bin/bash
#SBATCH --job-name=einsum_real
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:10:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/einsum_real_%j.log

set -e
export PYTHONUNBUFFERED=1

uv run --project /project/engvimmune/gleeai/boltz python3 - <<'PYEOF'
import torch
torch.manual_seed(42)
device = "cuda"

# REAL sizes from our pentamer
B, S, N, c_hidden = 1, 256, 2595, 32
N_half = 1298  # actual N//2 for 2-GPU
chunk_size = 4  # what Boltz2 uses

print("=" * 90)
print("OPM EINSUM — REAL SIZES (N=2595 vs N_half=1298)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 90)

# Generate random a and b in bf16 (same as actual model)
a_bf16 = torch.randn(B, S, N, c_hidden, device=device, dtype=torch.bfloat16)
b_bf16 = torch.randn(B, S, N, c_hidden, device=device, dtype=torch.bfloat16)

print(f"\n{'Test':<40} | {'mean_diff':>12} {'max_diff':>12} {'%differ':>10} {'match':>8}")
print("-" * 90)

# Test each c_hidden chunk separately (avoids OOM)
for ci in range(0, c_hidden, chunk_size):
    cs = min(chunk_size, c_hidden - ci)
    a_chunk = a_bf16[:, :, :, ci:ci+cs]

    # Full N rows einsum
    z_full = torch.einsum("bsic,bsjd->bijcd", a_chunk, b_bf16)  # [B, N, N, cs, c_h]
    # Take only first N_half rows
    z_full_half = z_full[:, :N_half, :, :, :].reshape(B, N_half, N, -1).float()

    del z_full
    torch.cuda.empty_cache()

    # Half N rows einsum (what DAP rank 0 sees)
    z_half = torch.einsum("bsic,bsjd->bijcd", a_chunk[:, :, :N_half, :], b_bf16)
    z_half = z_half.reshape(B, N_half, N, -1).float()

    diff = (z_full_half - z_half).abs()
    n_diff = (diff > 0).sum().item()
    pct = 100.0 * n_diff / diff.numel()
    ident = "✅" if n_diff == 0 else "❌"

    print(f"  chunk c_hidden[{ci}:{ci+cs}]                  | {diff.mean().item():>12.8f} {diff.max().item():>12.6f} {pct:>9.2f}% {ident:>8}")

    del z_full_half, z_half, diff
    torch.cuda.empty_cache()

# Also test the num_mask computation
print(f"\n{'num_mask comparison':<40}")
print("-" * 90)

mask = torch.randint(0, 2, (B, S, N), device=device, dtype=torch.bfloat16).unsqueeze(-1)  # [B,S,N,1]
mask_half = mask[:, :, :N_half, :]  # [B,S,N_half,1]

# Original OPM num_mask (full N x N)
for i in range(0, S, 64):
    chunk_m = mask[:, i:i+64, :, :]  # [B, 64, N, 1]
    cross = chunk_m[:, :, None, :, :] * chunk_m[:, :, :, None, :]  # [B,64,1,N,1] * [B,64,N,1,1] -> [B,64,N,N,1]
    if i == 0:
        num_full = cross.sum(1)
    else:
        num_full += cross.sum(1)
    del cross

# Scattered OPM num_mask (N_half x N)
for i in range(0, S, 64):
    chunk_ma = mask_half[:, i:i+64, :, :]  # [B, 64, N_half, 1]
    chunk_mb = mask[:, i:i+64, :, :]       # [B, 64, N, 1]
    cross = chunk_ma[:, :, :, None, :] * chunk_mb[:, :, None, :, :]  # [B,64,N_half,N,1]
    if i == 0:
        num_scat = cross.sum(1)
    else:
        num_scat += cross.sum(1)
    del cross

# Compare num_full[:, :N_half, :, :] vs num_scat
num_full_half = num_full[:, :N_half, :, :].float()
num_scat_f = num_scat.float()
diff_mask = (num_full_half - num_scat_f).abs()
n_diff_m = (diff_mask > 0).sum().item()
pct_m = 100.0 * n_diff_m / diff_mask.numel()
ident_m = "✅" if n_diff_m == 0 else "❌"
print(f"  num_mask (original vs scattered)       | {diff_mask.mean().item():>12.8f} {diff_mask.max().item():>12.6f} {pct_m:>9.2f}% {ident_m:>8}")

print("\nDONE")
PYEOF
