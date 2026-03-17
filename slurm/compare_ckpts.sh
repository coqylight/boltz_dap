#!/bin/bash
#SBATCH --job-name=compare_ckpts
#SBATCH --partition=normal
#SBATCH --account=engvimmune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/compare_ckpts_%j.log

BOLTZ_DIR=/project/engvimmune/gleeai/boltz
BASELINE=/project/engvimmune/gleeai/boltz_output/baseline_samedata_trunk_seed42/trunk_checkpoints.pt
DAP=/project/engvimmune/gleeai/boltz_output/dap_samedata_trunk_seed42/trunk_checkpoints.pt

echo "=============================================="
echo "Comparing trunk checkpoints (z/s)"
echo "=============================================="
uv run --project $BOLTZ_DIR python /project/engvimmune/gleeai/boltz_dap/scripts/compare_trunk_lazy.py \
    "$BASELINE" "$DAP"

echo ""
echo "=============================================="
echo "Comparing granular checkpoints"
echo "=============================================="
BASELINE_GRAN=/project/engvimmune/gleeai/boltz_output/baseline_samedata_trunk_seed42/granular_ckpts.pt
DAP_GRAN=/project/engvimmune/gleeai/boltz_output/dap_samedata_trunk_seed42/granular_ckpts.pt

if [ -f "$BASELINE_GRAN" ] && [ -f "$DAP_GRAN" ]; then
    uv run --project $BOLTZ_DIR python /project/engvimmune/gleeai/boltz_dap/scripts/compare_trunk_lazy.py \
        "$BASELINE_GRAN" "$DAP_GRAN"
else
    echo "Granular checkpoint files not found, skipping."
    ls -la "$BASELINE_GRAN" "$DAP_GRAN" 2>&1
fi

echo ""
echo "DONE"
