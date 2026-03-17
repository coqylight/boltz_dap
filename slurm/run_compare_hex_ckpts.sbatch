#!/bin/bash
#SBATCH --job-name=cmp_ckpt
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/compare_hex_ckpts_%j.log

export PYTHONUNBUFFERED=1

echo "=============================================="
echo "Comparing hexamer trunk z/s checkpoints"
echo "=============================================="

uv run --project /project/engvimmune/gleeai/boltz python3 -c "
import torch, gc

d = '/project/engvimmune/gleeai/boltz_output/orig_hex_trunk_ckpts'

pairs = [
    ('B_no_kernels', 'no_kernels.pt', 'C_kernels_tf32', 'kernels_tf32.pt'),
    ('B_no_kernels', 'no_kernels.pt', 'D_kernels_ieee', 'kernels_ieee.pt'),
    ('C_kernels_tf32', 'kernels_tf32.pt', 'D_kernels_ieee', 'kernels_ieee.pt'),
]

for name_a, file_a, name_b, file_b in pairs:
    print(f'\n{name_a} vs {name_b}:')
    a = torch.load(f'{d}/{file_a}', weights_only=True, map_location='cpu')
    b = torch.load(f'{d}/{file_b}', weights_only=True, map_location='cpu')
    for key in sorted(a.keys()):
        if key not in b:
            continue
        t_a = a[key].float()
        t_b = b[key].float()
        diff = (t_a - t_b).abs()
        match = torch.equal(t_a, t_b)
        print(f'  {key}: identical={match} mean={diff.mean():.8f} max={diff.max():.8f}')
    del a, b
    gc.collect()

print('\nDONE')
"

echo "=============================================="
echo "ALL DONE"
echo "=============================================="
