#!/bin/bash
#SBATCH --job-name=compare_gr
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/compare_granular_%j.log

set -e

BOLTZ_DIR=/project/engvimmune/gleeai/boltz
DAP_DIR=/project/engvimmune/gleeai/boltz_dap
OUT_BASE=/project/engvimmune/gleeai/boltz_output
BASELINE_TRUNK_DIR=$OUT_BASE/baseline_samedata_trunk_seed42
DAP_TRUNK_DIR=$OUT_BASE/dap_samedata_trunk_seed42

cd $DAP_DIR

echo "=============================================="
echo "Compare GRANULAR template+MSA checkpoints"
echo "=============================================="
uv run --project $BOLTZ_DIR python compare_granular.py \
    $BASELINE_TRUNK_DIR/granular_ckpts.pt \
    $DAP_TRUNK_DIR/granular_ckpts.pt

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
