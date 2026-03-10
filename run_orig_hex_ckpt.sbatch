#!/bin/bash
#SBATCH --job-name=orig_hckpt
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/orig_hex_ckpt_%j.log

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

INPUT="/project/engvimmune/gleeai/boltz_input/1LP3_hexamer.yaml"
CKPT_DIR="/project/engvimmune/gleeai/boltz_output/orig_hex_trunk_ckpts"
mkdir -p "$CKPT_DIR"

echo "=============================================="
echo "Original Boltz2 HEXAMER — 3 configs with z/s checkpointing"
echo "NOTE: These will likely OOM, but may save partial z/s before crashing"
echo "=============================================="

# -----------------------------------------------
# B: No kernels
# -----------------------------------------------
echo ""
echo "=== RUN B: No kernels ==="
BOLTZ_SAVE_TRUNK_CKPT="$CKPT_DIR/no_kernels.pt" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/orig_hex_ckpt_no_kernels \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --no_kernels \
    --override || echo "  ⚠ Run B failed (likely OOM), checking partial ckpts..."

echo "  Checkpoint exists: $(ls -la $CKPT_DIR/no_kernels.pt 2>/dev/null || echo 'NO')"

# -----------------------------------------------
# C: Kernels DEFAULT (TF32)
# -----------------------------------------------
echo ""
echo "=== RUN C: Kernels DEFAULT (TF32) ==="
BOLTZ_KERNEL_PRECISION=default BOLTZ_SAVE_TRUNK_CKPT="$CKPT_DIR/kernels_tf32.pt" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/orig_hex_ckpt_kernels_tf32 \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override || echo "  ⚠ Run C failed (likely OOM), checking partial ckpts..."

echo "  Checkpoint exists: $(ls -la $CKPT_DIR/kernels_tf32.pt 2>/dev/null || echo 'NO')"

# -----------------------------------------------
# D: Kernels IEEE f32
# -----------------------------------------------
echo ""
echo "=== RUN D: Kernels IEEE f32 ==="
BOLTZ_KERNEL_PRECISION=ieee BOLTZ_SAVE_TRUNK_CKPT="$CKPT_DIR/kernels_ieee.pt" \
uv run --project /project/engvimmune/gleeai/boltz \
    boltz predict "$INPUT" \
    --out_dir /project/engvimmune/gleeai/boltz_output/orig_hex_ckpt_kernels_ieee \
    --use_msa_server \
    --recycling_steps 3 \
    --sampling_steps 200 \
    --diffusion_samples 1 \
    --override || echo "  ⚠ Run D failed (likely OOM), checking partial ckpts..."

echo "  Checkpoint exists: $(ls -la $CKPT_DIR/kernels_ieee.pt 2>/dev/null || echo 'NO')"

# -----------------------------------------------
# Compare whatever z/s was saved
# -----------------------------------------------
echo ""
echo "=============================================="
echo "Comparing trunk z/s checkpoints (if any)"
echo "=============================================="
python3 -c "
import torch, os, sys

ckpt_dir = '$CKPT_DIR'
files = {}
for name, fname in [('B_no_kernels', 'no_kernels.pt'), ('C_kernels_tf32', 'kernels_tf32.pt'), ('D_kernels_ieee', 'kernels_ieee.pt')]:
    path = os.path.join(ckpt_dir, fname)
    if os.path.exists(path):
        files[name] = torch.load(path, weights_only=True)
        print(f'{name}: {list(files[name].keys())}')
    else:
        print(f'{name}: NO CHECKPOINT (OOM before any recycling step completed)')

if len(files) < 2:
    print('\nNot enough checkpoints to compare')
    sys.exit(0)

# Find common keys
names = list(files.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        common = set(files[a].keys()) & set(files[b].keys())
        if not common:
            print(f'\n{a} vs {b}: no common keys')
            continue
        print(f'\n{a} vs {b}:')
        for key in sorted(common):
            t_a = files[a][key].float()
            t_b = files[b][key].float()
            diff = (t_a - t_b).abs()
            match = torch.equal(t_a, t_b)
            print(f'  {key}: identical={match} mean_diff={diff.mean():.8f} max_diff={diff.max():.8f}')

print('\nDONE')
"
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
