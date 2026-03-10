#!/bin/bash
#SBATCH --job-name=subop_cmp
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/subop_compare_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
SUBOP_BASE="/project/engvimmune/gleeai/boltz_output/subop_ckpts"
mkdir -p "$SUBOP_BASE"

echo "=============================================="
echo "Sub-Op Divergence Test: Pentamer (N=2595)"
echo "=============================================="

# -----------------------------------------------
# 1. Original Boltz2 — No Kernels
# -----------------------------------------------
echo ""
echo "=== ORIG: No kernels ==="
BOLTZ_SAVE_SUBOP_CKPT="$SUBOP_BASE/orig_no_kernels" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/subop_orig_no_kernels \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --no_kernels \
    --override
echo "  ✓ Orig no-kernels done"

# -----------------------------------------------
# 2. Original Boltz2 — Kernels IEEE
# -----------------------------------------------
echo ""
echo "=== ORIG: Kernels IEEE ==="
BOLTZ_KERNEL_PRECISION=ieee BOLTZ_SAVE_SUBOP_CKPT="$SUBOP_BASE/orig_kernels_ieee" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/subop_orig_kernels_ieee \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override
echo "  ✓ Orig kernels-ieee done"

# -----------------------------------------------
# 3. DAP-2 — No Kernels
# -----------------------------------------------
echo ""
echo "=== DAP-2: No kernels ==="
BOLTZ_SAVE_SUBOP_CKPT="$SUBOP_BASE/dap_no_kernels" \
srun uv run --project /project/engvimmune/gleeai/boltz torchrun --nproc_per_node=2 \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/subop_dap_no_kernels \
    --no_kernels \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1
echo "  ✓ DAP no-kernels done"

# -----------------------------------------------
# 4. DAP-2 — Kernels IEEE
# -----------------------------------------------
echo ""
echo "=== DAP-2: Kernels IEEE ==="
BOLTZ_KERNEL_PRECISION=ieee BOLTZ_SAVE_SUBOP_CKPT="$SUBOP_BASE/dap_kernels_ieee" \
srun uv run --project /project/engvimmune/gleeai/boltz torchrun --nproc_per_node=2 \
    --master_port 29501 \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/subop_dap_kernels_ieee \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1
echo "  ✓ DAP kernels-ieee done"

# -----------------------------------------------
# 5. Compare sub-op z checkpoints
# -----------------------------------------------
echo ""
echo "=============================================="
echo "Comparing sub-op z checkpoints"
echo "=============================================="

python3 -c "
import torch
import os

base = '$SUBOP_BASE'
subops = ['after_trimul_out', 'after_trimul_in', 'after_triatt_start', 'after_triatt_end']
configs = {
    'orig_no_kernels': os.path.join(base, 'orig_no_kernels'),
    'orig_kernels_ieee': os.path.join(base, 'orig_kernels_ieee'),
    'dap_no_kernels': os.path.join(base, 'dap_no_kernels'),
    'dap_kernels_ieee': os.path.join(base, 'dap_kernels_ieee'),
}

# Verify files exist
for name, d in configs.items():
    print(f'{name}: ', end='')
    found = [s for s in subops if os.path.exists(os.path.join(d, s + '.pt'))]
    print(f'{len(found)}/{len(subops)} files found ({found})')
print()

# Load all data
data = {}
for name, d in configs.items():
    data[name] = {}
    for s in subops:
        f = os.path.join(d, s + '.pt')
        if os.path.exists(f):
            data[name][s] = torch.load(f, weights_only=True, map_location='cpu').float()

# === Key comparison: DAP vs Orig (same config) at each sub-op ===
print('='*60)
print('DAP vs ORIG — per sub-op z diff (SAME kernel config)')
print('='*60)
for tag in ['no_kernels', 'kernels_ieee']:
    dap_key = f'dap_{tag}'
    orig_key = f'orig_{tag}'
    print(f'\\n--- {dap_key} vs {orig_key} ---')
    for s in subops:
        if s in data.get(dap_key, {}) and s in data.get(orig_key, {}):
            d = data[dap_key][s]
            o = data[orig_key][s]
            # Trim to min shape in case of padding
            n = min(d.shape[1], o.shape[1])
            d = d[:, :n, :n, :]
            o = o[:, :n, :n, :]
            diff = (d - o).abs()
            print(f'  {s:25s}: mean={diff.mean():10.6f}  max={diff.max():10.4f}')
        else:
            print(f'  {s:25s}: MISSING')

# === Cross-config comparison: no-kernels vs kernels-ieee within DAP ===
print()
print('='*60)
print('Within DAP — no_kernels vs kernels_ieee per sub-op')
print('='*60)
for s in subops:
    if s in data.get('dap_no_kernels', {}) and s in data.get('dap_kernels_ieee', {}):
        a = data['dap_no_kernels'][s]
        b = data['dap_kernels_ieee'][s]
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n, :n, :]
        b = b[:, :n, :n, :]
        diff = (a - b).abs()
        print(f'  {s:25s}: mean={diff.mean():10.6f}  max={diff.max():10.4f}')
    else:
        print(f'  {s:25s}: MISSING')

# === Cross-config comparison: no-kernels vs kernels-ieee within ORIG ===
print()
print('='*60)
print('Within ORIG — no_kernels vs kernels_ieee per sub-op')
print('='*60)
for s in subops:
    if s in data.get('orig_no_kernels', {}) and s in data.get('orig_kernels_ieee', {}):
        a = data['orig_no_kernels'][s]
        b = data['orig_kernels_ieee'][s]
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n, :n, :]
        b = b[:, :n, :n, :]
        diff = (a - b).abs()
        print(f'  {s:25s}: mean={diff.mean():10.6f}  max={diff.max():10.4f}')
    else:
        print(f'  {s:25s}: MISSING')

print()
print('ALL COMPARISONS DONE')
"

echo "=============================================="
echo "COMPLETE"
echo "=============================================="
