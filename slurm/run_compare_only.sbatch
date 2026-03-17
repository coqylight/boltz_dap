#!/bin/bash
#SBATCH --job-name=compare_zs
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/compare_zs_%j.log

set -e

BOLTZ_DIR=/project/engvimmune/gleeai/boltz
DAP_DIR=/project/engvimmune/gleeai/boltz_dap
OUT_BASE=/project/engvimmune/gleeai/boltz_output
BASELINE_TRUNK_DIR=$OUT_BASE/baseline_samedata_trunk_seed42
DAP_TRUNK_DIR=$OUT_BASE/dap_samedata_trunk_seed42

cd $DAP_DIR

echo "=============================================="
echo "STEP 6: Compare trunk checkpoints (z/s)"
echo "=============================================="
uv run --project $BOLTZ_DIR python "${DAP_DIR}/scripts/compare_checkpoints.py" \
    $BASELINE_TRUNK_DIR/trunk_checkpoints.pt \
    $DAP_TRUNK_DIR/trunk_checkpoints.pt

echo ""
echo "=============================================="
echo "STEP 7: Compare GRANULAR template+MSA checkpoints"
echo "=============================================="
if [ -f "$BASELINE_TRUNK_DIR/granular_ckpts.pt" ] && [ -f "$DAP_TRUNK_DIR/granular_ckpts.pt" ]; then
    uv run --project $BOLTZ_DIR python "${DAP_DIR}/scripts/compare_granular.py" \
        $BASELINE_TRUNK_DIR/granular_ckpts.pt \
        $DAP_TRUNK_DIR/granular_ckpts.pt
else
    echo "WARNING: granular_ckpts.pt not found in one or both dirs"
    ls -la $BASELINE_TRUNK_DIR/granular_ckpts.pt 2>/dev/null || echo "  Missing: $BASELINE_TRUNK_DIR/granular_ckpts.pt"
    ls -la $DAP_TRUNK_DIR/granular_ckpts.pt 2>/dev/null || echo "  Missing: $DAP_TRUNK_DIR/granular_ckpts.pt"
fi

echo ""
echo "=============================================="
echo "MEMORY COMPARISON"
echo "=============================================="
echo "Baseline: OOM at 74,710 MB (single GPU)"
echo "DAP peak (from job log):"
grep 'GPU.*Peak' /project/engvimmune/gleeai/boltz_output/dap_samedata_test_346194.log 2>/dev/null || echo "  (check job log manually)"

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
