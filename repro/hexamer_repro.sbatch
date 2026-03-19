#!/bin/bash
#SBATCH --job-name=boltz-dap-repro
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.log

set -euo pipefail

# Required:
#   INPUT_YAML=/absolute/path/to/1LP3_hexamer_from_trimer_fill.yaml
#   BOLTZ_DIR=/absolute/path/to/boltz
#
# Optional:
#   REPO_DIR=/absolute/path/to/boltz_dap      (default: current working directory)
#   OUT_DIR=/absolute/path/to/output_dir      (default: $REPO_DIR/repro/output/hexamer_af3_defaults)
#   CACHE_DIR=/absolute/path/to/.boltz        (default: ~/.boltz)
#   SEED=42                                   (default: unset; non-deterministic)
#   USE_MSA_SERVER=1                          (default: 1)

REPO_DIR="${REPO_DIR:-$PWD}"
BOLTZ_DIR="${BOLTZ_DIR:?Set BOLTZ_DIR=/absolute/path/to/boltz}"
INPUT_YAML="${INPUT_YAML:?Set INPUT_YAML=/absolute/path/to/hexamer_input.yaml}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/repro/output/hexamer_af3_defaults}"
CACHE_DIR="${CACHE_DIR:-$HOME/.boltz}"
USE_MSA_SERVER="${USE_MSA_SERVER:-1}"
SEED="${SEED:-}"

DAP_SCRIPT="$REPO_DIR/boltz_dap_v2/run_boltz_dap_v2.py"
CHAIN_SCRIPT="$REPO_DIR/scripts/summarize_chain_pair_metrics.py"
REPORT_SCRIPT="$REPO_DIR/repro/collect_hexamer_repro_report.py"
VENV_ACTIVATE="$BOLTZ_DIR/.venv/bin/activate"

source "$VENV_ACTIVATE"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Boltz-DAP Hexamer Reproduction"
echo "  Job: $SLURM_JOB_ID  Node: $HOSTNAME"
echo "  Started: $(date)"
echo "============================================================"
echo "REPO_DIR=$REPO_DIR"
echo "BOLTZ_DIR=$BOLTZ_DIR"
echo "INPUT_YAML=$INPUT_YAML"
echo "OUT_DIR=$OUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "USE_MSA_SERVER=$USE_MSA_SERVER"
if [ -n "$SEED" ]; then
  echo "SEED=$SEED"
else
  echo "SEED=<unset>"
fi
echo ""

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BOLTZ_OUT_DIR="$OUT_DIR"

EXTRA_ARGS=(
  --recycling_steps 10
  --sampling_steps 200
  --diffusion_samples 25
  --use_potentials
)

if [ "$USE_MSA_SERVER" = "1" ]; then
  EXTRA_ARGS+=(--use_msa_server)
fi

if [ -n "$SEED" ]; then
  EXTRA_ARGS+=(--seed "$SEED")
fi

DAP_PORT=$((29700 + RANDOM % 100))
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --rdzv-endpoint=localhost:${DAP_PORT} \
  "$DAP_SCRIPT" \
  "$INPUT_YAML" \
  --out_dir "$OUT_DIR" \
  --cache "$CACHE_DIR" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "[post] Generating chain-pair tables..."
python "$CHAIN_SCRIPT" --out_dir "$OUT_DIR"

echo "[post] Generating markdown report..."
python "$REPORT_SCRIPT" --out_dir "$OUT_DIR"

echo ""
echo "DONE — $(date)"
