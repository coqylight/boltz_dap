#!/bin/bash
#SBATCH --job-name=boltz_orig
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/boltz_orig_pentamer_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="/project/engvimmune/gleeai/boltz_output/boltz_orig_pentamer"
INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"

echo "=============================================="
echo "Original Boltz2 predict — 1LP3 pentamer"
echo "Input: $INPUT"
echo "Output: $OUT_DIR"
echo "=============================================="

uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir "$OUT_DIR" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1

echo ""
echo "=============================================="
echo "DONE"
echo "=============================================="
