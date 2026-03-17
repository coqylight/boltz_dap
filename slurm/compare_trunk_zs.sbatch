#!/bin/bash
#SBATCH --job-name=cmp_zs
#SBATCH --account=engvimmune
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:10:00
#SBATCH --output=/project/engvimmune/gleeai/boltz_output/compare_zs_%j.log

set -e
export PYTHONUNBUFFERED=1

uv run --project /project/engvimmune/gleeai/boltz python3 - <<'PYEOF'
import torch

# v2 results from job 346322 (use_kernels=True except template PF)
ref_path = "/project/engvimmune/gleeai/boltz_output/ref_subop_v2/trunk_checkpoints.pt"
dap_path = "/project/engvimmune/gleeai/boltz_output/dap_subop_v2/trunk_checkpoints.pt"

print(f"Reference: {ref_path}")
print(f"DAP:       {dap_path}")
print()

ref = torch.load(ref_path, map_location="cpu", weights_only=True)
dap = torch.load(dap_path, map_location="cpu", weights_only=True)

print(f"Ref keys: {sorted(ref.keys())}")
print(f"DAP keys: {sorted(dap.keys())}")
print()

# Compare z and s from each recycle checkpoint
print("=" * 120)
print(f"{'Key':<45} {'tensor':<6} {'identical':>10} {'mean_diff':>12} {'max_diff':>12} {'shape'}")
print("-" * 120)

for key in sorted(ref.keys()):
    if key not in dap:
        print(f"{key:<45} MISSING in DAP")
        continue
    r = ref[key]
    d = dap[key]
    if isinstance(r, dict):
        for subk in ['z', 's']:
            if subk not in r:
                continue
            rv = r[subk].float()
            dv = d[subk].float()
            # Trim to match shapes
            for dim in range(rv.dim()):
                if rv.shape[dim] != dv.shape[dim]:
                    ms = min(rv.shape[dim], dv.shape[dim])
                    rv = rv.narrow(dim, 0, ms)
                    dv = dv.narrow(dim, 0, ms)
            diff = (rv - dv).abs()
            ident = "✅" if diff.max().item() == 0 else "❌"
            n_diff = (diff > 0).sum().item()
            pct = 100.0 * n_diff / diff.numel() if diff.numel() > 0 else 0
            print(f"{key:<45} {subk:<6} {ident:>10} {diff.mean():>12.8f} {diff.max():>12.8f} {list(rv.shape)}  ({pct:.2f}% differ)")

# Also compare granular checkpoints
print()
print("=" * 120)
print("GRANULAR CHECKPOINT COMPARISON (v2)")
print("=" * 120)

ref_g = torch.load("/project/engvimmune/gleeai/boltz_output/ref_subop_v2/granular_ckpts.pt", map_location="cpu", weights_only=True)
dap_g = torch.load("/project/engvimmune/gleeai/boltz_output/dap_subop_v2/granular_ckpts.pt", map_location="cpu", weights_only=True)

ordered = [
    "tmpl/a_tij", "tmpl/z_proj_out", "tmpl/v_input",
    "tmpl/pf0/input", "tmpl/pf0/after_tri_mul_out", "tmpl/pf0/after_tri_mul_in",
    "tmpl/pf0/after_tri_att_start", "tmpl/pf0/after_tri_att_end", "tmpl/pf0/after_transition",
    "tmpl/v_after_pf0",
    "tmpl/pf1/input", "tmpl/pf1/after_tri_mul_out", "tmpl/pf1/after_tri_mul_in",
    "tmpl/pf1/after_tri_att_start", "tmpl/pf1/after_tri_att_end", "tmpl/pf1/after_transition",
    "tmpl/v_after_pf1",
    "tmpl/v_residual", "tmpl/v_norm", "tmpl/u_agg", "tmpl/u_proj", "tmpl/z_final",
    "msa/blk0/after_opm", "msa/blk0/after_pf", "msa/z_out_residual",
]
common = sorted(set(ref_g.keys()) & set(dap_g.keys()))
for k in common:
    if k not in ordered:
        ordered.append(k)
ordered_common = [k for k in ordered if k in common]

print(f"\n{'Key':<45}| {'identical':>10} {'diff_mean':>12} {'diff_max':>12} {'n_diff':>12} {'n_total':>12}")
print("-" * 120)

for key in ordered_common:
    r = ref_g[key].float()
    d = dap_g[key].float()
    for dim in range(r.dim()):
        if r.shape[dim] != d.shape[dim]:
            ms = min(r.shape[dim], d.shape[dim])
            r = r.narrow(dim, 0, ms)
            d = d.narrow(dim, 0, ms)
    diff = (r - d).abs()
    ident = "✅" if diff.max().item() == 0 else "❌"
    n_diff = (diff > 0).sum().item()
    print(f"{key:<45}| {ident:>10} {diff.mean():>12.8f} {diff.max():>12.8f} {n_diff:>12} {diff.numel():>12}")

PYEOF

echo ""
echo "DONE"
