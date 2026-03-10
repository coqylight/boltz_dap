#!/bin/bash
#SBATCH --job-name=subop_dap
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/subop_dap_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer.yaml"
SUBOP_BASE="/project/engvimmune/gleeai/boltz_output/subop_ckpts"

# Original checkpoints already saved from job 346690:
#   $SUBOP_BASE/orig_no_kernels/  (4 files)
#   $SUBOP_BASE/orig_kernels_ieee/  (4 files)

echo "=============================================="
echo "Sub-Op DAP Runs Only (orig already done)"
echo "=============================================="

# -----------------------------------------------
# 1. DAP-2 — No Kernels
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
# 2. DAP-2 — Kernels IEEE
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
# 3. Compare all sub-op z checkpoints
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
    found = [s for s in subops if os.path.exists(os.path.join(d, s + '.pt'))]
    print(f'{name}: {len(found)}/{len(subops)} files')
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
    prev_diff_mean = 0.0
    for s in subops:
        if s in data.get(dap_key, {}) and s in data.get(orig_key, {}):
            d = data[dap_key][s]
            o = data[orig_key][s]
            n = min(d.shape[1], o.shape[1])
            d = d[:, :n, :n, :]
            o = o[:, :n, :n, :]
            diff = (d - o).abs()
            curr = diff.mean().item()
            delta = curr - prev_diff_mean
            print(f'  {s:25s}: mean={curr:10.6f}  max={diff.max():10.4f}  (delta=+{delta:.6f})')
            prev_diff_mean = curr
        else:
            print(f'  {s:25s}: MISSING')

# === Cross-config within DAP ===
print()
print('='*60)
print('Within DAP — no_kernels vs kernels_ieee per sub-op')
print('='*60)
prev_diff_mean = 0.0
for s in subops:
    if s in data.get('dap_no_kernels', {}) and s in data.get('dap_kernels_ieee', {}):
        a = data['dap_no_kernels'][s]
        b = data['dap_kernels_ieee'][s]
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n, :n, :]
        b = b[:, :n, :n, :]
        diff = (a - b).abs()
        curr = diff.mean().item()
        delta = curr - prev_diff_mean
        print(f'  {s:25s}: mean={curr:10.6f}  max={diff.max():10.4f}  (delta=+{delta:.6f})')
        prev_diff_mean = curr
    else:
        print(f'  {s:25s}: MISSING')

# === Cross-config within ORIG ===
print()
print('='*60)
print('Within ORIG — no_kernels vs kernels_ieee per sub-op')
print('='*60)
prev_diff_mean = 0.0
for s in subops:
    if s in data.get('orig_no_kernels', {}) and s in data.get('orig_kernels_ieee', {}):
        a = data['orig_no_kernels'][s]
        b = data['orig_kernels_ieee'][s]
        n = min(a.shape[1], b.shape[1])
        a = a[:, :n, :n, :]
        b = b[:, :n, :n, :]
        diff = (a - b).abs()
        curr = diff.mean().item()
        delta = curr - prev_diff_mean
        print(f'  {s:25s}: mean={curr:10.6f}  max={diff.max():10.4f}  (delta=+{delta:.6f})')
        prev_diff_mean = curr
    else:
        print(f'  {s:25s}: MISSING')

print()
print('ALL COMPARISONS DONE')
"

echo "=============================================="
echo "COMPLETE"
echo "=============================================="
