#!/bin/bash
#SBATCH --job-name=DAP_v4_tmpl_debug_and_plddt
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=28
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/slurm_v4_tmpl_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/slurm_v4_tmpl_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

# ──────────────────────────────────────────────────────────────
# Investigation job: Template sub-op comparison + full inference pLDDT
# Runs 3 steps:
#   1. Baseline trunk with template sub-op stats (1 GPU)
#   2. DAP trunk with template sub-op stats (2 GPUs)
#   3. Original Boltz2 full inference for pLDDT baseline (1 GPU)
# ──────────────────────────────────────────────────────────────

set -e

BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASELINE_OUT="/project/engvimmune/gleeai/boltz_output/baseline_tmpl_debug_seed42"
DAP_OUT="/project/engvimmune/gleeai/boltz_output/dap_tmpl_debug_seed42"
ORIG_PREDICT="/project/engvimmune/gleeai/boltz_output/original_predict_seed42"
SEED=42

echo "============================================================"
echo "TEMPLATE SUB-OP INVESTIGATION + FULL INFERENCE COMPARISON"
echo "$(date)"
echo "============================================================"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Step 1: Baseline trunk with template sub-op stats (1 GPU) ──
echo ""
echo "============================================================"
echo "[STEP 1/4] Baseline trunk with template sub-op stats..."
echo "============================================================"
cd "$DAP_DIR"
CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    python run_trunk_only.py "$INPUT" \
    --out_dir "$BASELINE_OUT" \
    --seed "$SEED" \
    --use_msa_server

echo ""
echo "  ✓ Baseline trunk done."
ls -lh "$BASELINE_OUT/trunk_checkpoints.pt" 2>/dev/null || echo "  WARNING: checkpoint file not found"

# ── Step 2: DAP trunk with template sub-op stats (2 GPUs) ──
echo ""
echo "============================================================"
echo "[STEP 2/4] DAP trunk with template sub-op stats (2 GPUs)..."
echo "============================================================"
cd "$DAP_DIR"
uv run --project "$BOLTZ_PROJECT" \
    torchrun --nproc_per_node=2 --standalone boltz_dap_v2/run_boltz_dap_v2.py \
    "$INPUT" \
    --out_dir "$DAP_OUT" \
    --seed "$SEED" \
    --use_msa_server

echo ""
echo "  ✓ DAP trunk done."
ls -lh "$DAP_OUT/trunk_checkpoints.pt" 2>/dev/null || echo "  WARNING: checkpoint file not found"

# ── Step 3: Compare checkpoints ──
echo ""
echo "============================================================"
echo "[STEP 3/4] Comparing checkpoints..."
echo "============================================================"
cd "$DAP_DIR"
uv run --project "$BOLTZ_PROJECT" python compare_checkpoints.py \
    "$BASELINE_OUT/trunk_checkpoints.pt" \
    "$DAP_OUT/trunk_checkpoints.pt"

# ── Step 4: Original Boltz2 full inference for pLDDT ──
echo ""
echo "============================================================"
echo "[STEP 4/4] Original Boltz2 full inference (pLDDT baseline)..."
echo "============================================================"
cd "$BOLTZ_PROJECT"
CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    boltz predict "$INPUT" \
    --out_dir "$ORIG_PREDICT" \
    --use_msa_server \
    --recycling_steps 3

echo ""
echo "  ✓ Original Boltz2 inference done."
echo "  DAP CIF: $(ls /project/engvimmune/gleeai/boltz_output/dap_v2fix3_seed42/predictions/1LP3*/1LP3*model_0.cif 2>/dev/null)"
echo "  Original CIF: $(ls $ORIG_PREDICT/predictions/1LP3*/1LP3*model_0.cif 2>/dev/null)"

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "============================================================"
echo ""
echo "=== TEMPLATE SUB-OP SUMMARY ==="
echo "Compare the [R0/TMPL] lines from Step 1 (baseline) vs Step 2 (DAP)"
echo "to identify the FIRST sub-op where stats diverge."
echo ""
echo "=== Key sub-ops to compare: ==="
echo "  a_tij       — template features (computed on full N, should match)"
echo "  v_input     — z_proj + a_tij merger (DAP scatters first)"
echo "  v_after_pf0 — after PF layer 0 (DAP uses DAPPairformerNoSeqLayer)"
echo "  v_after_pf1 — after PF layer 1 (same)"
echo "  v_residual  — after outer residual v + pf_output"
echo "  u_agg       — template aggregation"
echo "  u_proj      — final template output"
