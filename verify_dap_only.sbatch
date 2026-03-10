#!/bin/bash
#SBATCH --job-name=dap_verify
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_dap/verify_%j.log
#SBATCH --error=/project/engvimmune/gleeai/boltz_dap/verify_%j.log

set -e

VENV="/project/engvimmune/gleeai/boltz/.venv/bin/activate"
INPUT="/project/engvimmune/gleeai/kaggle/yamls/9QZJ.yaml"
DAP_OUT="/project/engvimmune/gleeai/boltz_dap/verify_output/dap"
DAP_SCRIPT="/project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py"
BASELINE_CIF="/project/engvimmune/gleeai/kaggle/predictions/boltz_results_9QZJ/predictions/9QZJ/9QZJ_model_0.cif"
CACHE="$HOME/.boltz"

source "$VENV"

echo "=========================================="
echo " DAP Fix Verification — 9QZJ (19 nt)"
echo " DAP only — baseline from existing run"
echo "=========================================="

# ── Run DAP Boltz2 on 2 GPUs ──
echo ""
echo ">>> Running DAP Boltz2 (2 GPUs)"
mkdir -p "$DAP_OUT"

torchrun --nproc_per_node=2 \
    "$DAP_SCRIPT" \
    "$INPUT" \
    --out_dir "$DAP_OUT" \
    --cache "$CACHE" \
    --diffusion_samples 1

echo "DAP done."
echo "DAP CIF files:"
find "$DAP_OUT" -name "*.cif" | sort

# ── Compare with existing baseline ──
echo ""
echo ">>> Comparing DAP vs baseline"
python3 << 'PYEOF'
import sys, glob
import gemmi
import numpy as np

def extract_coords(cif_path):
    st = gemmi.read_structure(cif_path)
    model = st[0]
    coords = []
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.name == "C1'":
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords)

baseline_cif = "/project/engvimmune/gleeai/kaggle/predictions/boltz_results_9QZJ/predictions/9QZJ/9QZJ_model_0.cif"
dap_cifs = sorted(glob.glob('/project/engvimmune/gleeai/boltz_dap/verify_output/dap/**/9QZJ*model_0.cif', recursive=True))

if not dap_cifs:
    print("ERROR: No DAP CIF found")
    sys.exit(1)

dap_cif = dap_cifs[0]
print(f"Baseline CIF: {baseline_cif}")
print(f"DAP CIF:      {dap_cif}")

bc = extract_coords(baseline_cif)
dc = extract_coords(dap_cif)

print(f"\nBaseline residues: {len(bc)}")
print(f"DAP residues:      {len(dc)}")

if len(bc) != len(dc):
    print("ERROR: Different number of residues!")
    sys.exit(1)

diff = np.abs(bc - dc)
rmsd = np.sqrt(np.mean((bc - dc) ** 2))

print(f"\nCoordinate comparison:")
print(f"  Max absolute diff: {diff.max():.6f} Å")
print(f"  Mean absolute diff: {diff.mean():.6f} Å")
print(f"  RMSD: {rmsd:.6f} Å")

if diff.max() < 0.1:
    print("\n✅ PASS — DAP matches baseline (< 0.1 Å)")
elif diff.max() < 1.0:
    print("\n⚠️  CLOSE — nearly matches (< 1.0 Å, bf16 diffs expected)")
else:
    print("\n❌ FAIL — diverged from baseline (> 1.0 Å)")
PYEOF

echo ""
echo "=========================================="
echo " Verification complete"
echo "=========================================="
