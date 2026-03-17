#!/bin/bash
#SBATCH --job-name=hex-dap4
#SBATCH --account=engvimmune
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/hex_dap4_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/hex_dap4_%j.log
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

set -e

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_hexamer.yaml"
BASE_OUT="/project/engvimmune/gleeai/boltz_output"
CKPT_BASE="${BASE_OUT}/v2_hex_recycle_ckpts"

DAP_SCRIPT="/project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py"
VENV="/project/engvimmune/gleeai/boltz/.venv/bin/activate"
CACHE="$HOME/.boltz"

source "$VENV"

echo "=========================================="
echo "  Hexamer DAP-4 — Per-Recycling Checkpoints"
echo "  4 GPUs to avoid OOM during z gather"
echo "=========================================="

# ===== DAP-4 — no kernels =====
echo ""
echo "=== [1/3] DAP-4 no-kernels ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_nk" \
_BOLTZ_RECYCLE_CTR=0 \
  torchrun --nproc_per_node=4 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/v2_hex_dap4_nk" \
    --cache "$CACHE" \
    --use_msa_server \
    --no_kernels

# ===== DAP-4 — kernels IEEE =====
echo ""
echo "=== [2/3] DAP-4 kernels-IEEE ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_ieee" \
_BOLTZ_RECYCLE_CTR=0 \
BOLTZ_KERNEL_PRECISION=ieee \
  torchrun --nproc_per_node=4 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/v2_hex_dap4_ieee" \
    --cache "$CACHE" \
    --use_msa_server

# ===== DAP-4 — kernels TF32 =====
echo ""
echo "=== [3/3] DAP-4 kernels-TF32 ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_tf32" \
_BOLTZ_RECYCLE_CTR=0 \
  torchrun --nproc_per_node=4 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/v2_hex_dap4_tf32" \
    --cache "$CACHE" \
    --use_msa_server

# ===== Compare with existing original data =====
echo ""
echo "=========================================="
echo "  Comparing: Hexamer Orig vs DAP-4"
echo "=========================================="

python3 - << 'PYEOF'
import torch, os, glob, gc

ckpt_base = "/project/engvimmune/gleeai/boltz_output/v2_hex_recycle_ckpts"

def get_recycle_files(dirpath):
    return sorted(glob.glob(os.path.join(dirpath, "recycle_*.pt")))

pairs = [
    ("orig_nk",   "dap_nk",   "No-Kernels"),
    ("orig_ieee", "dap_ieee", "Kernels-IEEE"),
    ("orig_tf32", "dap_tf32", "Kernels-TF32"),
]

print("=" * 70)
print("  Hexamer: Per-Recycling-Step (Orig vs DAP-4, same config)")
print("=" * 70)

for orig_dir, dap_dir, label in pairs:
    orig_files = get_recycle_files(os.path.join(ckpt_base, orig_dir))
    dap_files = get_recycle_files(os.path.join(ckpt_base, dap_dir))

    print(f"\n  {label}")
    print(f"  orig files: {[os.path.basename(f) for f in orig_files]}")
    print(f"  dap  files: {[os.path.basename(f) for f in dap_files]}")

    n = min(len(orig_files), len(dap_files))
    for i in range(n):
        z_orig = torch.load(orig_files[i], map_location="cpu", weights_only=True).float()
        z_dap = torch.load(dap_files[i], map_location="cpu", weights_only=True).float()
        diff = (z_orig - z_dap).abs()
        print(f"  R{i}: z_mean={diff.mean():.6f}  z_max={diff.max():.4f}")
        del z_orig, z_dap, diff; gc.collect()

print("\nDONE")
PYEOF
