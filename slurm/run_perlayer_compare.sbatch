#!/bin/bash
#SBATCH --job-name=perlayer-cmp
#SBATCH --account=engvimmune
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/perlayer_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/perlayer_%j.log
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

set -e

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
BASE_OUT="/project/engvimmune/gleeai/boltz_output"
CKPT_BASE="${BASE_OUT}/perlayer_ckpts"
CACHE="$HOME/.boltz"

BOLTZ_DIR="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
DAP_SCRIPT="${DAP_DIR}/boltz_dap_v2/run_boltz_dap_v2.py"
VENV="${BOLTZ_DIR}/.venv/bin/activate"

source "$VENV"

echo "=========================================="
echo "  Per-Layer Divergence Experiment"
echo "  Pentamer N=2595, 6 configs"
echo "=========================================="

# ===== 1. Original Boltz2 — no kernels =====
echo ""
echo "=== [1/6] Original no-kernels ==="
BOLTZ_SAVE_LAYER_CKPT="${CKPT_BASE}/orig_no_kernels" \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/perlayer_orig_nk" \
    --no_kernels \
    --recycling_steps 1 \
    --use_msa_server \
    --num_workers 4

# ===== 2. Original Boltz2 — kernels IEEE =====
echo ""
echo "=== [2/6] Original kernels-IEEE ==="
BOLTZ_SAVE_LAYER_CKPT="${CKPT_BASE}/orig_kernels_ieee" \
BOLTZ_KERNEL_PRECISION=ieee \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/perlayer_orig_ieee" \
    --recycling_steps 1 \
    --use_msa_server \
    --num_workers 4

# ===== 3. Original Boltz2 — kernels TF32 =====
echo ""
echo "=== [3/6] Original kernels-TF32 ==="
BOLTZ_SAVE_LAYER_CKPT="${CKPT_BASE}/orig_kernels_tf32" \
CUDA_VISIBLE_DEVICES=0 \
  boltz predict "$INPUT" \
    --out_dir "${BASE_OUT}/perlayer_orig_tf32" \
    --recycling_steps 1 \
    --use_msa_server \
    --num_workers 4

# ===== 4. DAP-2 — no kernels =====
echo ""
echo "=== [4/6] DAP-2 no-kernels ==="
BOLTZ_SAVE_LAYER_CKPT="${CKPT_BASE}/dap_no_kernels" \
  torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/perlayer_dap_nk" \
    --cache "$CACHE" \
    --use_msa_server \
    --no_kernels \
    --recycling_steps 1

# ===== 5. DAP-2 — kernels IEEE =====
echo ""
echo "=== [5/6] DAP-2 kernels-IEEE ==="
BOLTZ_SAVE_LAYER_CKPT="${CKPT_BASE}/dap_kernels_ieee" \
BOLTZ_KERNEL_PRECISION=ieee \
  torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/perlayer_dap_ieee" \
    --cache "$CACHE" \
    --use_msa_server \
    --recycling_steps 1

# ===== 6. DAP-2 — kernels TF32 =====
echo ""
echo "=== [6/6] DAP-2 kernels-TF32 ==="
BOLTZ_SAVE_LAYER_CKPT="${CKPT_BASE}/dap_kernels_tf32" \
  torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "${BASE_OUT}/perlayer_dap_tf32" \
    --cache "$CACHE" \
    --use_msa_server \
    --recycling_steps 1

# ===== Compare all 3 pairs =====
echo ""
echo "=========================================="
echo "  Comparing per-layer z divergence"
echo "=========================================="

python3 - << 'PYEOF'
import torch
import os
import glob

ckpt_base = "/project/engvimmune/gleeai/boltz_output/perlayer_ckpts"

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

    # Find common layer files
    orig_files = sorted(glob.glob(os.path.join(orig_path, "layer_*.pt")))
    dap_files = sorted(glob.glob(os.path.join(dap_path, "layer_*.pt")))
    orig_names = {os.path.basename(f) for f in orig_files}
    dap_names = {os.path.basename(f) for f in dap_files}
    common = sorted(orig_names & dap_names)

    print(f"\n{'='*60}")
    print(f"  {label}: Original vs DAP-2")
    print(f"{'='*60}")
    print(f"  {'Layer':>8}  {'Mean Diff':>12}  {'Max Diff':>12}  {'Std Diff':>12}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*12}")

    for name in common:
        z_orig = torch.load(os.path.join(orig_path, name), map_location="cpu", weights_only=True)
        z_dap = torch.load(os.path.join(dap_path, name), map_location="cpu", weights_only=True)
        diff = (z_orig.float() - z_dap.float()).abs()
        layer_num = int(name.replace("layer_", "").replace(".pt", ""))
        print(f"  {layer_num:>8}  {diff.mean().item():>12.6f}  {diff.max().item():>12.4f}  {diff.std().item():>12.6f}")
        del z_orig, z_dap, diff

print("\nDONE")
PYEOF
