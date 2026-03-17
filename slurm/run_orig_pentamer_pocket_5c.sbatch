#!/bin/bash
#SBATCH --job-name=pent_pkt5c
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/pentamer_pocket_5c_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/pentamer_pocket_5c_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="/project/engvimmune/gleeai/boltz_output/pentamer_pocket_5c"
INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_pocket_5c.yaml"

echo "=============================================="
echo "Original Boltz2 — Pentamer pocket (5 constraints, 6.1-side only)"
echo "Input: $INPUT"
echo "Output: $OUT_DIR"
echo "Started: $(date)"
echo "=============================================="

uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir "$OUT_DIR" \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1

echo ""
echo "Exit code: $?"
echo ""

CIF=$(find "$OUT_DIR" -name "*.cif" -type f 2>/dev/null | head -1)
if [ -n "$CIF" ]; then
    echo "  ✅ CIF file: $CIF"
else
    echo "  ❌ No CIF file found"
fi

echo ""
echo "DONE — $(date)"
