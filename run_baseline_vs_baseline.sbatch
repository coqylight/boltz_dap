#!/bin/bash
#SBATCH --job-name=base_vs_base
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/baseline_vs_baseline_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

set -e

LOG_FILE=/project/engvimmune/gleeai/boltz_output/baseline_vs_baseline.log
exec > >(tee "$LOG_FILE") 2>&1

BOLTZ_DIR=/project/engvimmune/gleeai/boltz
DAP_DIR=/project/engvimmune/gleeai/boltz_dap
INPUT_YAML=/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml
OUT_BASE=/project/engvimmune/gleeai/boltz_output

# Use same processed data for both runs
SOURCE_DIR=$OUT_BASE/baseline_subops_seed42
RUN_A=$OUT_BASE/baseline_runA_seed42
RUN_B=$OUT_BASE/baseline_runB_seed42

echo "=============================================="
echo "BASELINE vs BASELINE: Proving CUDA non-determinism"
echo "=============================================="
echo ""
echo "If run A vs run B shows same ~0.000008 diffs as DAP vs baseline,"
echo "it proves those diffs are CUDA noise, NOT from DAP logic."
echo ""

echo "=============================================="
echo "STEP 1: Copy same processed data to both dirs"
echo "=============================================="
rm -rf $RUN_A $RUN_B
mkdir -p $RUN_A $RUN_B
cp -r $SOURCE_DIR/processed $RUN_A/
cp -r $SOURCE_DIR/processed $RUN_B/
echo "Done."

echo ""
echo "=============================================="
echo "STEP 2: Run baseline trunk - RUN A"
echo "=============================================="
cd $DAP_DIR
CUDA_VISIBLE_DEVICES=0 uv run --project $BOLTZ_DIR python run_trunk_only.py \
    $INPUT_YAML \
    --out_dir $RUN_A \
    --seed 42

echo ""
echo "=============================================="
echo "STEP 3: Run baseline trunk - RUN B (identical)"
echo "=============================================="
CUDA_VISIBLE_DEVICES=0 uv run --project $BOLTZ_DIR python run_trunk_only.py \
    $INPUT_YAML \
    --out_dir $RUN_B \
    --seed 42

echo ""
echo "=============================================="
echo "STEP 4: Compare RUN A vs RUN B"
echo "=============================================="
echo "If these show ~0.000008 diff: CUDA non-determinism confirmed."
echo "If these show 0.000000 diff: DAP has a real bug."
echo ""
uv run --project $BOLTZ_DIR python compare_checkpoints.py \
    $RUN_A/trunk_checkpoints.pt \
    $RUN_B/trunk_checkpoints.pt

echo ""
echo "=============================================="
echo "STEP 5: Compare with DAP run (from previous job)"
echo "=============================================="
echo "For reference: DAP vs baseline diffs"
uv run --project $BOLTZ_DIR python compare_checkpoints.py \
    $RUN_A/trunk_checkpoints.pt \
    $OUT_BASE/dap_samedata_trunk_seed42/trunk_checkpoints.pt

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
