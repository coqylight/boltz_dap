#!/bin/bash
#SBATCH --job-name=hex-dockDE
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/hex_dockDE_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/hex_dockDE_%j.log

echo "============================================================"
echo "  Hexamer DAP-4: Chain F docked between Chain D and E"
echo "  Job: $SLURM_JOB_ID  Node: $HOSTNAME"
echo "  Started: $(date)"
echo "============================================================"

BOLTZ_DIR="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
DAP_SCRIPT="${DAP_DIR}/boltz_dap_v2/run_boltz_dap_v2.py"
VENV="${BOLTZ_DIR}/.venv/bin/activate"
CACHE="$HOME/.boltz"

source "$VENV"

INPUT=/project/engvimmune/gleeai/boltz_input/1LP3_hexamer_dock_DE.yaml
OUT_DIR=/project/engvimmune/gleeai/boltz_output/hex_dockDE

rm -rf "$OUT_DIR"

echo ""
echo "  Input: $INPUT"
echo "  Output: $OUT_DIR"
echo "  DAP size: 4"
echo ""

DAP_PORT=$((29700 + RANDOM % 100))
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --rdzv-endpoint=localhost:${DAP_PORT} \
  "$DAP_SCRIPT" \
  "$INPUT" \
  --out_dir "$OUT_DIR" \
  --cache "$CACHE" \
  --use_msa_server \
  --seed 42

echo ""
echo "  Exit code: $?"
echo ""

CIF=$(find "$OUT_DIR" -name "*.cif" -type f 2>/dev/null | head -1)
if [ -n "$CIF" ]; then
    echo "  ✅ CIF file: $CIF"
else
    echo "  ❌ No CIF file found"
fi

echo ""
echo "DONE — $(date)"
