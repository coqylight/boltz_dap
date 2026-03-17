#!/bin/bash
# Sub-op diagnosis v3: BOTH runs with --no_kernels to eliminate cuequivariance differences.
# This should produce bitwise-identical results for all sub-ops.
set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

YAML="/project/engvimmune/gleeai/boltz_input/1LP3_pentamer_from_tetramer.yaml"
BASE="/project/engvimmune/gleeai/boltz_output"
REF_OUT="$BASE/ref_subop_v3"
DAP_OUT="$BASE/dap_subop_v3"
BOLTZ_PROJECT="/project/engvimmune/gleeai/boltz"
SEED=42

PROCESSED_SRC="$BASE/dap_predict_seed42/processed"
if [ ! -d "$PROCESSED_SRC" ]; then
    PROCESSED_SRC="$BASE/baseline_predict_seed42/boltz_results_1LP3_pentamer_from_tetramer/processed"
fi

echo "=============================================="
echo "STEP 1: Run 1-GPU reference (--no_kernels)"
echo "=============================================="
rm -rf "$REF_OUT"
mkdir -p "$REF_OUT"
cp -r "$PROCESSED_SRC" "$REF_OUT/processed"

CUDA_VISIBLE_DEVICES=0 uv run --project "$BOLTZ_PROJECT" \
    torchrun --nproc_per_node=1 --standalone \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$REF_OUT" \
    --seed $SEED \
    --no_kernels \
    --use_msa_server

echo ""
echo "=============================================="
echo "STEP 2: Run 2-GPU DAP (--no_kernels)"
echo "=============================================="
rm -rf "$DAP_OUT"
mkdir -p "$DAP_OUT"
cp -r "$PROCESSED_SRC" "$DAP_OUT/processed"

NCCL_TIMEOUT=7200 uv run --project "$BOLTZ_PROJECT" \
    torchrun --nproc_per_node=2 --standalone \
    /project/engvimmune/gleeai/boltz_dap/boltz_dap_v2/run_boltz_dap_v2.py \
    "$YAML" \
    --out_dir "$DAP_OUT" \
    --seed $SEED \
    --no_kernels \
    --use_msa_server

echo ""
echo "=============================================="
echo "STEP 3: Compare granular sub-op checkpoints"
echo "=============================================="

uv run --project "$BOLTZ_PROJECT" python3 - <<'PYEOF'
import torch

ref_path = "/project/engvimmune/gleeai/boltz_output/ref_subop_v3/granular_ckpts.pt"
dap_path = "/project/engvimmune/gleeai/boltz_output/dap_subop_v3/granular_ckpts.pt"

print(f"Reference: {ref_path}")
print(f"DAP:       {dap_path}")

ref = torch.load(ref_path, map_location="cpu", weights_only=True)
dap = torch.load(dap_path, map_location="cpu", weights_only=True)

ref_keys = sorted(ref.keys())
dap_keys = sorted(dap.keys())
common = sorted(set(ref_keys) & set(dap_keys))

print(f"\nRef keys: {len(ref_keys)}, DAP keys: {len(dap_keys)}, Common: {len(common)}")
if set(ref_keys) - set(dap_keys):
    print(f"  Only in ref: {sorted(set(ref_keys) - set(dap_keys))}")
if set(dap_keys) - set(ref_keys):
    print(f"  Only in DAP: {sorted(set(dap_keys) - set(ref_keys))}")

# Order: template sub-ops, then MSA sub-ops
ordered = [
    "tmpl/a_tij", "tmpl/z_proj_out", "tmpl/v_input",
    "tmpl/pf0/input", "tmpl/pf0/after_tri_mul_out", "tmpl/pf0/after_tri_mul_in",
    "tmpl/pf0/after_tri_att_start", "tmpl/pf0/after_tri_att_end", "tmpl/pf0/after_transition",
    "tmpl/v_after_pf0",
    "tmpl/pf1/input", "tmpl/pf1/after_tri_mul_out", "tmpl/pf1/after_tri_mul_in",
    "tmpl/pf1/after_tri_att_start", "tmpl/pf1/after_tri_att_end", "tmpl/pf1/after_transition",
    "tmpl/v_after_pf1",
    "tmpl/v_residual", "tmpl/v_norm", "tmpl/u_agg", "tmpl/u_proj", "tmpl/z_final",
    # MSA sub-ops (new finer checkpoints)
    "msa/blk0/after_pwa_and_transition_m",
    "msa/blk0/before_opm_z",
    "msa/blk0/after_opm",
    "msa/blk0/after_pf",
    "msa/z_out_residual",
]
for k in common:
    if k not in ordered:
        ordered.append(k)

ordered_common = [k for k in ordered if k in common]

print(f"\n{'='*130}")
print(f"{'Key':<45}| {'identical':>10} {'diff_mean':>12} {'diff_max':>12} {'n_diff':>12} {'n_total':>12}")
print("-" * 130)

first_div = None
for key in ordered_common:
    r = ref[key].float()
    d = dap[key].float()
    shapes_match = True
    for dim in range(r.dim()):
        if r.shape[dim] != d.shape[dim]:
            min_s = min(r.shape[dim], d.shape[dim])
            r = r.narrow(dim, 0, min_s)
            d = d.narrow(dim, 0, min_s)
            shapes_match = False
    
    identical = torch.equal(ref[key], dap[key]) if shapes_match else (r - d).abs().max().item() == 0
    diff = (r - d).abs()
    n_diff = (diff > 0).sum().item()
    mark = "✅" if identical else "❌"
    if not identical and first_div is None:
        first_div = key
    print(f"{key:<45}| {mark:>10} {diff.mean():>12.8f} {diff.max():>12.8f} {n_diff:>12} {diff.numel():>12}")

print(f"\n{'='*130}")
if first_div:
    print(f"FIRST DIVERGENCE: {first_div}")
    r = ref[first_div].float()
    d = dap[first_div].float()
    for dim in range(r.dim()):
        min_s = min(r.shape[dim], d.shape[dim])
        r = r.narrow(dim, 0, min_s)
        d = d.narrow(dim, 0, min_s)
    diff = (r - d).abs()
    print(f"  Elements differing: {(diff > 0).sum().item()} / {diff.numel()}")
    print(f"  Mean diff:   {diff.mean():.10f}")
    print(f"  Max diff:    {diff.max():.10f}")
    idx = ordered_common.index(first_div)
    if idx > 0:
        prev_key = ordered_common[idx - 1]
        r_p = ref[prev_key].float()
        d_p = dap[prev_key].float()
        for dim in range(r_p.dim()):
            min_s = min(r_p.shape[dim], d_p.shape[dim])
            r_p = r_p.narrow(dim, 0, min_s)
            d_p = d_p.narrow(dim, 0, min_s)
        print(f"  Previous '{prev_key}': {'✅ identical' if (r_p - d_p).abs().max().item() == 0 else '❌ diverged'}")
else:
    print("ALL SUB-OPS BITWISE IDENTICAL! ✅✅✅")

PYEOF

echo ""
echo "=============================================="
echo "ALL DONE"
echo "=============================================="
