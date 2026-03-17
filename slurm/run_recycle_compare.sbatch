#!/bin/bash
#SBATCH --job-name=recycle-cmp
#SBATCH --account=engvimmune
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/recycle_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/recycle_%j.log
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
CKPT_BASE="${BASE_OUT}/recycle_ckpts"

BOLTZ_DIR="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
DAP_SCRIPT="${DAP_DIR}/boltz_dap_v2/run_boltz_dap_v2.py"
VENV="${BOLTZ_DIR}/.venv/bin/activate"
CACHE="$HOME/.boltz"

source "$VENV"

echo "=========================================="
echo "  Per-Recycling-Step Divergence Experiment"
echo "  Pentamer N=2595, 6 configs, default recycling"
echo "=========================================="

# ===== 1. Original Boltz2 — no kernels =====
echo ""
echo "=== [1/6] Original no-kernels ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/orig_no_kernels" \
_BOLTZ_RECYCLE_CTR=0 \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/recycle_orig_nk" \
    --no_kernels \
    --use_msa_server \
    --num_workers 4

# ===== 2. Original Boltz2 — kernels IEEE =====
echo ""
echo "=== [2/6] Original kernels-IEEE ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/orig_kernels_ieee" \
_BOLTZ_RECYCLE_CTR=0 \
BOLTZ_KERNEL_PRECISION=ieee \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/recycle_orig_ieee" \
    --use_msa_server \
    --num_workers 4

# ===== 3. Original Boltz2 — kernels TF32 =====
echo ""
echo "=== [3/6] Original kernels-TF32 ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/orig_kernels_tf32" \
_BOLTZ_RECYCLE_CTR=0 \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/recycle_orig_tf32" \
    --use_msa_server \
    --num_workers 4

# ===== 4. DAP-2 — no kernels =====
echo ""
echo "=== [4/6] DAP-2 no-kernels ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_no_kernels" \
_BOLTZ_RECYCLE_CTR=0 \
  torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/recycle_dap_nk" \
    --cache "$CACHE" \
    --use_msa_server \
    --no_kernels

# ===== 5. DAP-2 — kernels IEEE =====
echo ""
echo "=== [5/6] DAP-2 kernels-IEEE ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_kernels_ieee" \
_BOLTZ_RECYCLE_CTR=0 \
BOLTZ_KERNEL_PRECISION=ieee \
  torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/recycle_dap_ieee" \
    --cache "$CACHE" \
    --use_msa_server

# ===== 6. DAP-2 — kernels TF32 =====
echo ""
echo "=== [6/6] DAP-2 kernels-TF32 ==="
BOLTZ_SAVE_RECYCLE_CKPT="${CKPT_BASE}/dap_kernels_tf32" \
_BOLTZ_RECYCLE_CTR=0 \
  torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/recycle_dap_tf32" \
    --cache "$CACHE" \
    --use_msa_server

# ===== Compare all 3 pairs =====
echo ""
echo "=========================================="
echo "  Comparing per-recycling-step z divergence"
echo "=========================================="

python3 - << 'PYEOF'
import torch
import os
import glob

ckpt_base = "/project/engvimmune/gleeai/boltz_output/recycle_ckpts"

pairs = [
    ("orig_no_kernels",   "dap_no_kernels",   "No-Kernels"),
    ("orig_kernels_ieee", "dap_kernels_ieee",  "Kernels-IEEE"),
    ("orig_kernels_tf32", "dap_kernels_tf32",  "Kernels-TF32"),
]

for orig_dir, dap_dir, label in pairs:
    orig_path = os.path.join(ckpt_base, orig_dir)
    dap_path = os.path.join(ckpt_base, dap_dir)

    if not os.path.isdir(orig_path) or not os.path.isdir(dap_path):
        print(f"\n[{label}] SKIPPED — missing directory")
        continue

    orig_files = sorted(glob.glob(os.path.join(orig_path, "recycle_*.pt")))
    dap_files = sorted(glob.glob(os.path.join(dap_path, "recycle_*.pt")))
    orig_names = {os.path.basename(f) for f in orig_files}
    dap_names = {os.path.basename(f) for f in dap_files}
    common = sorted(orig_names & dap_names)

    print(f"\n{'='*60}")
    print(f"  {label}: Original vs DAP-2 (per recycling step)")
    print(f"{'='*60}")
    print(f"  {'Recycle':>8}  {'Mean Diff':>12}  {'Max Diff':>12}  {'Std Diff':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

    for name in common:
        z_orig = torch.load(os.path.join(orig_path, name), map_location="cpu", weights_only=True)
        z_dap = torch.load(os.path.join(dap_path, name), map_location="cpu", weights_only=True)
        diff = (z_orig.float() - z_dap.float()).abs()
        rc_num = int(name.replace("recycle_", "").replace(".pt", ""))
        print(f"  R{rc_num:>6}  {diff.mean().item():>12.6f}  {diff.max().item():>12.4f}  {diff.std().item():>12.6f}")
        del z_orig, z_dap, diff

print("\nDONE")
PYEOF
