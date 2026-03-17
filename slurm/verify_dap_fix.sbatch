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
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gleeai@connect.ust.hk

set -e

VENV="/project/engvimmune/gleeai/boltz/.venv/bin/activate"
INPUT="/project/engvimmune/gleeai/kaggle/yamls/9QZJ.yaml"
BASE_OUT="/project/engvimmune/gleeai/boltz_dap/verify_output"
DAP_SCRIPT="/project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py"
CACHE="$HOME/.boltz"

source "$VENV"

echo "=========================================="
echo " DAP Fix Verification — 9QZJ (19 nt)"
echo "=========================================="

# ── Step 1: Run original Boltz2 on GPU 0 (single GPU baseline) ──
echo ""
echo ">>> STEP 1: Original Boltz2 (single GPU)"
BASELINE_OUT="${BASE_OUT}/baseline"
mkdir -p "$BASELINE_OUT"

CUDA_VISIBLE_DEVICES=0 boltz predict "$INPUT" \
    --out_dir "$BASELINE_OUT" \
    --cache "$CACHE" \
    --diffusion_samples 1 \
    --override recycling_steps=0 \
    --accelerator gpu \
    --devices 1

echo "Baseline done."
echo "Baseline CIF files:"
find "$BASELINE_OUT" -name "*.cif" | sort

# ── Step 2: Run DAP Boltz2 on 2 GPUs ──
echo ""
echo ">>> STEP 2: DAP Boltz2 (2 GPUs)"
DAP_OUT="${BASE_OUT}/dap"
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

# ── Step 3: Compare CIF outputs ──
echo ""
echo ">>> STEP 3: Comparing outputs"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '/project/engvimmune/gleeai/boltz_dap/boltz_dap_v2')
import gemmi
import numpy as np

def extract_coords(cif_path):
    """Extract all C1' coordinates from CIF."""
    st = gemmi.read_structure(cif_path)
    model = st[0]
    coords = []
    for chain in model:
        for res in chain:
            for atom in res:
                if atom.name == "C1'":
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return np.array(coords)

import glob

baseline_cifs = sorted(glob.glob('/project/engvimmune/gleeai/boltz_dap/verify_output/baseline/**/9QZJ*model_0.cif', recursive=True))
dap_cifs = sorted(glob.glob('/project/engvimmune/gleeai/boltz_dap/verify_output/dap/**/9QZJ*model_0.cif', recursive=True))

if not baseline_cifs:
    print("ERROR: No baseline CIF found")
    sys.exit(1)
if not dap_cifs:
    print("ERROR: No DAP CIF found")
    sys.exit(1)

print(f"Baseline CIF: {baseline_cifs[0]}")
print(f"DAP CIF:      {dap_cifs[0]}")

baseline_coords = extract_coords(baseline_cifs[0])
dap_coords = extract_coords(dap_cifs[0])

print(f"\nBaseline residues: {len(baseline_coords)}")
print(f"DAP residues:      {len(dap_coords)}")

if len(baseline_coords) != len(dap_coords):
    print("ERROR: Different number of residues!")
    sys.exit(1)

# Compare coordinates
diff = np.abs(baseline_coords - dap_coords)
rmsd = np.sqrt(np.mean((baseline_coords - dap_coords) ** 2))

print(f"\nCoordinate comparison:")
print(f"  Max absolute diff: {diff.max():.6f} Å")
print(f"  Mean absolute diff: {diff.mean():.6f} Å")
print(f"  RMSD: {rmsd:.6f} Å")

# Note: bf16 precision means small diffs are expected
if diff.max() < 0.1:
    print("\n✅ PASS — DAP output matches baseline (< 0.1 Å max diff)")
elif diff.max() < 1.0:
    print("\n⚠️  CLOSE — DAP output nearly matches baseline (< 1.0 Å max diff)")
    print("   Small diffs expected from bf16 accumulation order differences")
else:
    print("\n❌ FAIL — DAP output diverges from baseline (> 1.0 Å max diff)")
    print("   Something is wrong with the scatter-first implementation")
PYEOF

echo ""
echo "=========================================="
echo " Verification complete"
echo "=========================================="
