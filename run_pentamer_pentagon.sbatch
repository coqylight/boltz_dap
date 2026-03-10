#!/bin/bash
#SBATCH --job-name=pent-pentagon
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/pentamer_pentagon_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/pentamer_pentagon_%j.log

echo "============================================================"
echo "  Pentamer Pentagon: 5-chain radial petal with contacts"
echo "  Job: $SLURM_JOB_ID  Node: $HOSTNAME"
echo "  Started: $(date)"
echo "============================================================"

BOLTZ_DIR="/project/engvimmune/gleeai/boltz"
VENV="${BOLTZ_DIR}/.venv/bin/activate"

source "$VENV"

INPUT=/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_pentagon.yaml
OUT_DIR=/project/engvimmune/gleeai/boltz_output/pentamer_pentagon

rm -rf "$OUT_DIR"

echo ""
echo "  Input: $INPUT"
echo "  Output: $OUT_DIR"
echo "  GPU: single (original Boltz2)"
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

uv run --project "$BOLTZ_DIR" \
  boltz predict "$INPUT" \
  --out_dir "$OUT_DIR" \
  --use_msa_server \
  --recycling_steps 3 \
  --sampling_steps 200 \
  --diffusion_samples 1

echo ""
echo "  Exit code: $?"
echo ""

CIF=$(find "$OUT_DIR" -name "*.cif" -type f 2>/dev/null | head -1)
if [ -n "$CIF" ]; then
    echo "  ✅ CIF file: $CIF"
    echo "  Size: $(du -h "$CIF" | cut -f1)"
else
    echo "  ❌ No CIF file found"
fi

echo ""
echo "DONE — $(date)"
