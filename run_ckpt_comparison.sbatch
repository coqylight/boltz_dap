#!/bin/bash
#SBATCH --job-name=boltz_v2fix3
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/slurm_v2fix3_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/slurm_v2fix3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

# ──────────────────────────────────────────────────────────────
# Verify fix: MSA outer residual + correct baseline (with templates)
# Re-runs BOTH baseline and DAP since baseline was previously broken
# (missing template_dir → zero template output)
# ──────────────────────────────────────────────────────────────

set -e

BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASELINE_OUT="/project/engvimmune/gleeai/boltz_output/baseline_ckpt_v3_seed42"
DAP_OUT="/project/engvimmune/gleeai/boltz_output/dap_v2fix3_seed42"
SEED=42

echo "============================================================"
echo "DAP VERIFICATION (v3): MSA residual fix + correct baseline"
echo "$(date)"
echo "============================================================"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 1: Baseline trunk (single GPU, with templates) ──
echo ""
echo "============================================================"
echo "[STEP 1/3] Running BASELINE trunk (single GPU, with templates)..."
echo "============================================================"
cd "$DAP_DIR"
CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    python run_trunk_only.py "$INPUT" \
    --out_dir "$BASELINE_OUT" \
    --seed "$SEED" \
    --use_msa_server

echo ""
echo "  ✓ Baseline done. Checkpoints at: $BASELINE_OUT/trunk_checkpoints.pt"
ls -lh "$BASELINE_OUT/trunk_checkpoints.pt" 2>/dev/null || echo "  WARNING: checkpoint file not found"

# ── Step 2: DAP trunk (2 GPUs, with MSA residual fix) ──
echo ""
echo "============================================================"
echo "[STEP 2/3] Running DAP trunk (2 GPUs) with MSA residual fix..."
echo "============================================================"
cd "$DAP_DIR"
uv run --project "$BOLTZ_PROJECT" \
    torchrun --nproc_per_node=2 --standalone boltz_dap_v2/run_boltz_dap_v2.py \
    "$INPUT" \
    --out_dir "$DAP_OUT" \
    --seed "$SEED" \
    --use_msa_server

echo ""
echo "  ✓ DAP done."
ls -lh "$DAP_OUT/trunk_checkpoints.pt" 2>/dev/null || echo "  WARNING: checkpoint file not found"

# ── Step 3: Compare checkpoints ──
echo ""
echo "============================================================"
echo "[STEP 3/3] Comparing checkpoints (baseline w/templates vs DAP)..."
echo "============================================================"
cd "$DAP_DIR"
uv run --project "$BOLTZ_PROJECT" python compare_checkpoints.py \
    "$BASELINE_OUT/trunk_checkpoints.pt" \
    "$DAP_OUT/trunk_checkpoints.pt"

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "============================================================"
