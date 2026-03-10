#!/bin/bash
#SBATCH --job-name=tmpl-shard-verify
#SBATCH --account=engvimmune
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/tmpl_shard_verify_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_output/tmpl_shard_verify_%j.log
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

set -e

# ============================================================
#  Template Pairformer Sharding Verification
#
#  Runs original Boltz2 (kernels-TF32, 1 GPU) and DAP-2/4
#  for 4 template-based inputs, compares z/s, collects CIF files.
# ============================================================

BOLTZ_DIR="/project/engvimmune/gleeai/boltz"
DAP_DIR="/project/engvimmune/gleeai/boltz_dap"
DAP_SCRIPT="${DAP_DIR}/boltz_dap_v2/run_boltz_dap_v2.py"
VENV="${BOLTZ_DIR}/.venv/bin/activate"
CACHE="$HOME/.boltz"
BASE_OUT="/project/engvimmune/gleeai/boltz_output/tmpl_shard_verify"
CIF_DIR="${BASE_OUT}/cif_files"

INPUT_DIR="/project/engvimmune/gleeai/boltz_input"

source "$VENV"
mkdir -p "$BASE_OUT" "$CIF_DIR"

echo "============================================================"
echo "  Template Pairformer Sharding Verification"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Date: $(date)"
echo "============================================================"

# Helper: collect CIF file from output dir
collect_cif() {
    local src_dir="$1"
    local dest_name="$2"
    local cif=$(find "${src_dir}" -name "*.cif" 2>/dev/null | head -1)
    if [ -n "$cif" ]; then
        cp "$cif" "${CIF_DIR}/${dest_name}.cif"
        echo "  → Collected CIF: ${dest_name}.cif"
    else
        echo "  ⚠ No CIF found in ${src_dir}/predictions"
    fi
}

# ===== Input definitions =====
# Each entry: (input_yaml, label, dap_gpus)
# dap_gpus: number of GPUs for DAP run
declare -a INPUTS=(
    "${INPUT_DIR}/1LP3_tetramer_from_trimer.yaml|tetra_from_tri|2"
    "${INPUT_DIR}/1LP3_pentamer_from_tetramer.yaml|penta_from_tetra|2"
    "${INPUT_DIR}/1LP3_hexamer_from_pentamer.yaml|hexa_from_penta|4"
    "${INPUT_DIR}/1LP3_tetramer.yaml|tetra_no_tmpl|2"
)

TOTAL=${#INPUTS[@]}
IDX=0

for entry in "${INPUTS[@]}"; do
    IFS='|' read -r INPUT LABEL DAP_GPUS <<< "$entry"
    IDX=$((IDX + 1))

    echo ""
    echo "============================================================"
    echo "  [$IDX/$TOTAL] ${LABEL}"
    echo "  Input: $(basename $INPUT)"
    echo "  DAP GPUs: ${DAP_GPUS}"
    echo "============================================================"

    ORIG_OUT="${BASE_OUT}/orig_${LABEL}"
    DAP_OUT="${BASE_OUT}/dap_${LABEL}"

    # --- Original Boltz2 (kernels-TF32, 1 GPU) ---
    echo ""
    echo "  --- Original Boltz2 (kernels-TF32) ---"
    MASTER_PORT=$((29700 + IDX)) \
    _BOLTZ_RECYCLE_CTR=0 \
    CUDA_VISIBLE_DEVICES=0 \
      boltz predict "$INPUT" \
        --out_dir "$ORIG_OUT" \
        --use_msa_server \
        --seed 42 \
        --num_workers 4 || {
        echo "  ⚠ Original run failed for ${LABEL} (may be OOM)"
    }

    collect_cif "$ORIG_OUT" "orig_${LABEL}"

    # Kill any leftover pytorch processes and wait for port release
    pkill -f "torch.distributed" 2>/dev/null || true
    sleep 5

    # --- DAP (kernels-TF32) ---
    echo ""
    echo "  --- DAP-${DAP_GPUS} (kernels-TF32, sharded template PF) ---"
    DAP_PORT=$((29600 + IDX))
    _BOLTZ_RECYCLE_CTR=0 \
      torchrun --nproc_per_node=${DAP_GPUS} --rdzv-endpoint=localhost:${DAP_PORT} \
        "$DAP_SCRIPT" \
        "$INPUT" \
        --out_dir "$DAP_OUT" \
        --cache "$CACHE" \
        --use_msa_server \
        --seed 42 || {
        echo "  ⚠ DAP run failed for ${LABEL}"
    }

    # Cleanup between runs
    pkill -f "torch.distributed" 2>/dev/null || true
    sleep 3

    collect_cif "$DAP_OUT" "dap_${LABEL}"

    echo ""
    echo "  ✓ ${LABEL} complete"
done

# ===== Generate memory timeline plots =====
echo ""
echo "============================================================"
echo "  Generating memory timeline plots"
echo "============================================================"

PLOT_SCRIPT="${DAP_DIR}/plot_memory.py"
DAP_DIRS=()
for entry in "${INPUTS[@]}"; do
    IFS='|' read -r INPUT LABEL DAP_GPUS <<< "$entry"
    DAP_OUT="${BASE_OUT}/dap_${LABEL}"
    if [ -f "${DAP_OUT}/mem_timeline.json" ]; then
        DAP_DIRS+=("$DAP_OUT")
        python3 "$PLOT_SCRIPT" "$DAP_OUT"
    fi
done

# Overlay all DAP timelines if multiple exist
if [ ${#DAP_DIRS[@]} -gt 1 ]; then
    python3 "$PLOT_SCRIPT" --overlay --out "${BASE_OUT}/mem_overlay_all.png" "${DAP_DIRS[@]}"
fi

# ===== Compare z/s tensors =====
echo ""
echo "============================================================"
echo "  Comparing z/s tensors: Original vs DAP"
echo "============================================================"

python3 - <<'PYEOF'
import torch
import os
import glob

base = "/project/engvimmune/gleeai/boltz_output/tmpl_shard_verify"

inputs = [
    ("tetra_from_tri",   "Tetramer from Trimer"),
    ("penta_from_tetra", "Pentamer from Tetramer"),
    ("hexa_from_penta",  "Hexamer from Pentamer"),
    ("tetra_no_tmpl",    "Tetramer (no template)"),
]

print(f"\n{'='*72}")
print(f"  Original vs DAP — z/s Tensor Comparison")
print(f"{'='*72}")
print(f"{'Input':>30} | {'z_mean':>10} | {'z_max':>10} | {'s_mean':>10} | {'s_max':>10}")
print("-" * 100)

results = {}
for label, desc in inputs:
    orig_zs = os.path.join(base, f"orig_{label}", "zs_tensors.pt")
    dap_zs = os.path.join(base, f"dap_{label}", "zs_tensors.pt")

    if not os.path.exists(orig_zs):
        print(f"{desc:>30} | SKIP (no orig zs_tensors.pt)")
        continue
    if not os.path.exists(dap_zs):
        print(f"{desc:>30} | SKIP (no dap zs_tensors.pt)")
        continue

    orig = torch.load(orig_zs, map_location="cpu", weights_only=True)
    dap = torch.load(dap_zs, map_location="cpu", weights_only=True)

    z_diff = (orig["z"].float() - dap["z"].float()).abs()
    s_diff = (orig["s"].float() - dap["s"].float()).abs()

    z_mean = z_diff.mean().item()
    z_max = z_diff.max().item()
    s_mean = s_diff.mean().item()
    s_max = s_diff.max().item()

    print(f"{desc:>30} | {z_mean:10.6f} | {z_max:10.4f} | {s_mean:10.6f} | {s_max:10.4f}")
    results[label] = {"z_mean": z_mean, "z_max": z_max, "s_mean": s_mean, "s_max": s_max}
    del orig, dap, z_diff, s_diff

# Summary
print(f"\n{'='*72}")
print(f"  Summary")
print(f"{'='*72}")
for label, desc in inputs:
    if label in results:
        r = results[label]
        status = "✓ PASS" if r["z_mean"] <= 0.15 else "✗ HIGH DIVERGENCE"
        print(f"  {desc:>30}: z_mean={r['z_mean']:.6f}  {status}")

# List CIF files
cif_dir = os.path.join(base, "cif_files")
if os.path.exists(cif_dir):
    cifs = sorted(os.listdir(cif_dir))
    print(f"\n  CIF files for ChimeraX comparison ({len(cifs)}):")
    for f in cifs:
        fpath = os.path.join(cif_dir, f)
        size_mb = os.path.getsize(fpath) / (1024*1024)
        print(f"    {f} ({size_mb:.1f} MB)")

# List memory timeline plots
mem_pngs = glob.glob(os.path.join(base, "**/mem_timeline.png"), recursive=True)
mem_pngs += glob.glob(os.path.join(base, "mem_overlay_all.png"))
if mem_pngs:
    print(f"\n  Memory timeline plots ({len(mem_pngs)}):")
    for p in mem_pngs:
        print(f"    {os.path.relpath(p, base)}")

print("\nDONE")
PYEOF

echo ""
echo "============================================================"
echo "  CIF files collected in: ${CIF_DIR}"
echo "  Copy to local machine for ChimeraX viewing:"
echo "    scp -r superpod:${CIF_DIR} ."
echo "============================================================"
echo "DONE — $(date)"
