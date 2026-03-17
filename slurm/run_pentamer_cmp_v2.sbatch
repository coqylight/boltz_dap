#!/bin/bash
#SBATCH --job-name=pent-cmp-v2
#SBATCH --account=engvimmune
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/pent_cmp_v2_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/pent_cmp_v2_%j.log
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

set -e

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
BASE_OUT="/project/engvimmune/gleeai/boltz_output"
CKPT_BASE="${BASE_OUT}/v2_recycle_ckpts"

BOLTZ_DIR="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
DAP_SCRIPT="${DAP_DIR}/boltz_dap_v2/run_boltz_dap_v2.py"
VENV="${BOLTZ_DIR}/.venv/bin/activate"
CACHE="$HOME/.boltz"

source "$VENV"

echo "=========================================="
echo "  Pentamer Original vs DAP — Full Comparison"
echo "  Default recycling, 6 configs"
echo "  Saves z at end of each recycling step"
echo "=========================================="

# ===== 1. Original Boltz2 — no kernels =====
echo ""
echo "=== [1/6] Original no-kernels ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/orig_nk" \
_BOLTZ_RECYCLE_CTR=0 \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/v2_orig_nk" \
    --no_kernels \
    --use_msa_server \
    --num_workers 4

# ===== 2. Original Boltz2 — kernels IEEE =====
echo ""
echo "=== [2/6] Original kernels-IEEE ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/orig_ieee" \
_BOLTZ_RECYCLE_CTR=0 \
BOLTZ_KERNEL_PRECISION=ieee \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/v2_orig_ieee" \
    --use_msa_server \
    --num_workers 4

# ===== 3. Original Boltz2 — kernels TF32 =====
echo ""
echo "=== [3/6] Original kernels-TF32 ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/orig_tf32" \
_BOLTZ_RECYCLE_CTR=0 \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/v2_orig_tf32" \
    --use_msa_server \
    --num_workers 4

# ===== 4. DAP-2 — no kernels =====
echo ""
echo "=== [4/6] DAP-2 no-kernels ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_nk" \
_BOLTZ_RECYCLE_CTR=0 \
  torchrun --nproc_per_node=2 --rdzv-endpoint=localhost:29501 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/v2_dap_nk" \
    --cache "$CACHE" \
    --use_msa_server \
    --no_kernels

# ===== 5. DAP-2 — kernels IEEE =====
echo ""
echo "=== [5/6] DAP-2 kernels-IEEE ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_ieee" \
_BOLTZ_RECYCLE_CTR=0 \
BOLTZ_KERNEL_PRECISION=ieee \
  torchrun --nproc_per_node=2 --rdzv-endpoint=localhost:29501 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/v2_dap_ieee" \
    --cache "$CACHE" \
    --use_msa_server

# ===== 6. DAP-2 — kernels TF32 =====
echo ""
echo "=== [6/6] DAP-2 kernels-TF32 ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_tf32" \
_BOLTZ_RECYCLE_CTR=0 \
  torchrun --nproc_per_node=2 --rdzv-endpoint=localhost:29501 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/v2_dap_tf32" \
    --cache "$CACHE" \
    --use_msa_server

# ===== Compare =====
echo ""
echo "=========================================="
echo "  Comparing final z: Original vs DAP"
echo "=========================================="

python3 - << 'PYEOF'
import torch
import os
import glob

ckpt_base = "/project/engvimmune/gleeai/boltz_output/v2_recycle_ckpts"

# For each config, find the LAST recycle file (= final recycling step output)
def get_last_recycle(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "recycle_*.pt")))
    if not files:
        return None
    return files[-1]

# ===== 1. Original vs DAP (same config) =====
print("=" * 70)
print("  Original vs DAP — Same Config (final recycling step)")
print("=" * 70)
print(f"{'Comparison':>35} | {'z_mean':>10} | {'z_max':>10}")
print("-" * 65)

pairs = [
    ("orig_nk",   "dap_nk",   "No-Kernels: Orig vs DAP"),
    ("orig_ieee", "dap_ieee", "Kernels-IEEE: Orig vs DAP"),
    ("orig_tf32", "dap_tf32", "Kernels-TF32: Orig vs DAP"),
]

for orig_dir, dap_dir, label in pairs:
    orig_file = get_last_recycle(os.path.join(ckpt_base, orig_dir))
    dap_file = get_last_recycle(os.path.join(ckpt_base, dap_dir))
    if not orig_file or not dap_file:
        print(f"{label:>35} | SKIPPED (missing)")
        continue
    z_orig = torch.load(orig_file, map_location="cpu", weights_only=True).float()
    z_dap = torch.load(dap_file, map_location="cpu", weights_only=True).float()
    diff = (z_orig - z_dap).abs()
    print(f"{label:>35} | {diff.mean():.6f} | {diff.max():>10.4f}")
    print(f"{'':>35}   orig={os.path.basename(orig_file)} dap={os.path.basename(dap_file)}")
    del z_orig, z_dap, diff

# ===== 2. Original configs cross-comparison =====
print()
print("=" * 70)
print("  Original Configs Cross-Comparison (final recycling step)")
print("=" * 70)
print(f"{'Comparison':>35} | {'z_mean':>10} | {'z_max':>10}")
print("-" * 65)

cross = [
    ("orig_nk",   "orig_ieee", "Orig nk vs Orig ieee"),
    ("orig_nk",   "orig_tf32", "Orig nk vs Orig tf32"),
    ("orig_ieee", "orig_tf32", "Orig ieee vs Orig tf32"),
]

for dir_a, dir_b, label in cross:
    fa = get_last_recycle(os.path.join(ckpt_base, dir_a))
    fb = get_last_recycle(os.path.join(ckpt_base, dir_b))
    if not fa or not fb:
        print(f"{label:>35} | SKIPPED (missing)")
        continue
    za = torch.load(fa, map_location="cpu", weights_only=True).float()
    zb = torch.load(fb, map_location="cpu", weights_only=True).float()
    diff = (za - zb).abs()
    print(f"{label:>35} | {diff.mean():.6f} | {diff.max():>10.4f}")
    del za, zb, diff

# ===== 3. DAP configs cross-comparison =====
print()
print("=" * 70)
print("  DAP Configs Cross-Comparison (final recycling step)")
print("=" * 70)
print(f"{'Comparison':>35} | {'z_mean':>10} | {'z_max':>10}")
print("-" * 65)

cross_dap = [
    ("dap_nk",   "dap_ieee", "DAP nk vs DAP ieee"),
    ("dap_nk",   "dap_tf32", "DAP nk vs DAP tf32"),
    ("dap_ieee", "dap_tf32", "DAP ieee vs DAP tf32"),
]

for dir_a, dir_b, label in cross_dap:
    fa = get_last_recycle(os.path.join(ckpt_base, dir_a))
    fb = get_last_recycle(os.path.join(ckpt_base, dir_b))
    if not fa or not fb:
        print(f"{label:>35} | SKIPPED (missing)")
        continue
    za = torch.load(fa, map_location="cpu", weights_only=True).float()
    zb = torch.load(fb, map_location="cpu", weights_only=True).float()
    diff = (za - zb).abs()
    print(f"{label:>35} | {diff.mean():.6f} | {diff.max():>10.4f}")
    del za, zb, diff

# ===== 4. Show per-recycling-step divergence =====
print()
print("=" * 70)
print("  Per-Recycling-Step Divergence (Orig vs DAP, same config)")
print("=" * 70)

for orig_dir, dap_dir, label in pairs:
    orig_path = os.path.join(ckpt_base, orig_dir)
    dap_path = os.path.join(ckpt_base, dap_dir)
    orig_files = sorted(glob.glob(os.path.join(orig_path, "recycle_*.pt")))
    dap_files = sorted(glob.glob(os.path.join(dap_path, "recycle_*.pt")))
    
    print(f"\n  {label}")
    print(f"  orig files: {[os.path.basename(f) for f in orig_files]}")
    print(f"  dap  files: {[os.path.basename(f) for f in dap_files]}")
    
    # Match by index
    n = min(len(orig_files), len(dap_files))
    for i in range(n):
        z_orig = torch.load(orig_files[i], map_location="cpu", weights_only=True).float()
        z_dap = torch.load(dap_files[i], map_location="cpu", weights_only=True).float()
        diff = (z_orig - z_dap).abs()
        print(f"  R{i}: z_mean={diff.mean():.6f}  z_max={diff.max():.4f}")
        del z_orig, z_dap, diff

print("\nDONE")
PYEOF
