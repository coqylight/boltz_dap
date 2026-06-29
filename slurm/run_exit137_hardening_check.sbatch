#!/bin/bash
#SBATCH --job-name=dap-exit137-check
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/exit137_hardening_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/exit137_hardening_%j.log

set -euo pipefail

# Usage:
#   sbatch slurm/run_exit137_hardening_check.sbatch
#   MODE=full sbatch slurm/run_exit137_hardening_check.sbatch
#
# MODE=smoke: 1 recycle, 1 diffusion sample (default; quick validation)
# MODE=full : 10 recycles, 5 diffusion samples (friend's failing condition)

REPO_DIR="${REPO_DIR:-/project/engvimmune/gleeai/boltz_dap_main_hardening}"
BOLTZ_DIR="${BOLTZ_DIR:-/project/engvimmune/gleeai/boltz}"
INPUT_YAML="${INPUT_YAML:-/project/engvimmune/gleeai/boltz_input/mutation_6_viability_bottom_20/row11093_false/verifying_template_boltz_template_force_false_row11093.yaml}"
SOURCE_OUT="${SOURCE_OUT:-/project/engvimmune/gleeai/boltz_output/verify_template_boltz/row11093_false/dap_4704011_clean}"
OUT_ROOT="${OUT_ROOT:-/project/engvimmune/gleeai/boltz_output/exit137_hardening_check}"
CACHE_DIR="${CACHE_DIR:-$HOME/.boltz}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MODE="${MODE:-smoke}"
SAMPLING_STEPS="${SAMPLING_STEPS:-200}"

case "$MODE" in
  smoke)
    RECYCLING_STEPS="${RECYCLING_STEPS:-1}"
    DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:-1}"
    ;;
  full)
    RECYCLING_STEPS="${RECYCLING_STEPS:-10}"
    DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES:-5}"
    ;;
  *)
    echo "ERROR: MODE must be 'smoke' or 'full' (got '$MODE')" >&2
    exit 2
    ;;
esac

OUT_DIR="${OUT_DIR:-$OUT_ROOT/row11093_false/${MODE}_${SLURM_JOB_ID}}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$BOLTZ_DIR/.venv_orig/bin/activate}"

source "$VENV_ACTIVATE"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo "  Boltz-DAP exit137 hardening check"
echo "  Job: $SLURM_JOB_ID  Node: $HOSTNAME"
echo "  Started: $(date)"
echo "============================================================"
echo "REPO_DIR=$REPO_DIR"
echo "BOLTZ_DIR=$BOLTZ_DIR"
echo "INPUT_YAML=$INPUT_YAML"
echo "SOURCE_OUT=$SOURCE_OUT"
echo "OUT_DIR=$OUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "MODE=$MODE"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "RECYCLING_STEPS=$RECYCLING_STEPS"
echo "SAMPLING_STEPS=$SAMPLING_STEPS"
echo "DIFFUSION_SAMPLES=$DIFFUSION_SAMPLES"
echo ""

echo "=== Hardening worktree status ==="
(
  cd "$REPO_DIR"
  git log -1 --oneline
  git status --short
)
echo ""

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BOLTZ_SAVE_TRUNK_CKPT=0
export BOLTZ_SAVE_GRAN_CKPT=0

ARGS=(
  "$INPUT_YAML"
  --out_dir "$OUT_DIR"
  --cache "$CACHE_DIR"
  --recycling_steps "$RECYCLING_STEPS"
  --sampling_steps "$SAMPLING_STEPS"
  --diffusion_samples "$DIFFUSION_SAMPLES"
  --no_save_trunk_checkpoints
  --no_save_granular_checkpoints
)

if [ -d "$SOURCE_OUT/processed" ]; then
  echo "Reusing processed inputs from: $SOURCE_OUT/processed"
  rm -rf "$OUT_DIR/processed"
  cp -a "$SOURCE_OUT/processed" "$OUT_DIR/processed"
  ARGS+=(--skip_processing)
else
  echo "WARNING: source processed dir not found; using MSA server."
  ARGS+=(--use_msa_server)
fi

echo ""
echo "=== Running ==="
printf ' %q' torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" "$REPO_DIR/boltz_dap_v2/run_boltz_dap_v2.py" "${ARGS[@]}"
echo ""

torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \
  "$REPO_DIR/boltz_dap_v2/run_boltz_dap_v2.py" \
  "${ARGS[@]}"

echo ""
echo "=== Key checks ==="
if grep -R "Trunk checkpoints: off | granular checkpoints: off" "$OUT_DIR" >/dev/null 2>&1; then
  echo "checkpoint banner found in output files"
else
  echo "checkpoint banner is in this Slurm log above"
fi
echo "Expected during trunk: [CKP] ... skipped full gather"
echo "Expected memory lines include: rss=... cgroup=..."
echo ""
echo "DONE - $(date)"
