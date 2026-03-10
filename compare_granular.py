#!/usr/bin/env python
"""
Compare granular template checkpoints between baseline and DAP.

Usage:
    python compare_granular.py /path/to/baseline/granular_ckpts.pt /path/to/dap/granular_ckpts.pt
"""

import sys
import torch


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <baseline_ckpts.pt> <dap_ckpts.pt>")
        sys.exit(1)

    base_path, dap_path = sys.argv[1], sys.argv[2]
    print(f"\n{'='*80}")
    print(f"  GRANULAR CHECKPOINT COMPARISON")
    print(f"{'='*80}")
    print(f"  Baseline: {base_path}")
    print(f"  DAP:      {dap_path}\n")

    base = torch.load(base_path, map_location="cpu", weights_only=True)
    dap = torch.load(dap_path, map_location="cpu", weights_only=True)

    # Find common keys, ordered logically
    all_keys = sorted(set(list(base.keys()) + list(dap.keys())))
    # Sort by logical order within template
    order = [
        "tmpl/a_tij", "tmpl/z_proj_out", "tmpl/v_input",
        # PF layer 0 sub-ops
        "tmpl/pf0/after_tri_mul_out", "tmpl/pf0/after_tri_mul_in",
        "tmpl/pf0/after_tri_att_start", "tmpl/pf0/after_tri_att_end",
        "tmpl/pf0/after_transition",
        "tmpl/v_after_pf0",
        # PF layer 1 sub-ops
        "tmpl/pf1/after_tri_mul_out", "tmpl/pf1/after_tri_mul_in",
        "tmpl/pf1/after_tri_att_start", "tmpl/pf1/after_tri_att_end",
        "tmpl/pf1/after_transition",
        "tmpl/v_after_pf1",
        # Post-PF
        "tmpl/v_residual", "tmpl/v_norm", "tmpl/u_agg", "tmpl/u_proj", "tmpl/z_final",
        # MSA block 0 sub-ops
        "msa/blk0/after_opm", "msa/blk0/after_pf",
        "msa/z_out_residual",
    ]
    ordered = [k for k in order if k in all_keys]
    remaining = [k for k in all_keys if k not in ordered]
    all_keys = ordered + remaining

    LW = 35  # label width
    print(f"  {'Label':<{LW}s} {'Shape':>20s} {'Mean Diff':>12s} {'Max Diff':>12s}  {'Status'}")
    print(f"  {'-'*LW} {'-'*20} {'-'*12} {'-'*12}  {'-'*20}")

    first_diff_key = None
    for key in all_keys:
        if key not in base:
            print(f"  {key:<{LW}s} {'(missing in baseline)':>20s}")
            continue
        if key not in dap:
            print(f"  {key:<{LW}s} {'(missing in DAP)':>20s}")
            continue

        bt = base[key].float()
        dt = dap[key].float()

        # Handle shape mismatch (e.g. v_input is [B*T, N, N, D] vs [B, T, N, N, D])
        if bt.shape != dt.shape:
            # Try reshaping to match
            if bt.numel() == dt.numel():
                dt = dt.reshape(bt.shape)
            else:
                print(f"  {key:<{LW}s} {'SHAPE MISMATCH':>20s}  base={list(bt.shape)} dap={list(dt.shape)}")
                continue

        diff = (bt - dt).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        if max_diff == 0:
            status = "✅ EXACT"
        elif max_diff < 0.01:
            status = f"⚠️  ~{max_diff:.4f} ULP"
            if first_diff_key is None:
                first_diff_key = key
        else:
            status = f"❌ LARGE"
            if first_diff_key is None:
                first_diff_key = key

        shape_str = "x".join(str(s) for s in bt.shape)
        print(f"  {key:<{LW}s} {shape_str:>20s} {mean_diff:>12.6f} {max_diff:>12.4f}  {status}")

    print()
    if first_diff_key:
        print(f"  ▶ FIRST DIVERGENCE AT: \"{first_diff_key}\"")

        # Detailed analysis of the first divergent checkpoint
        bt = base[first_diff_key].float()
        dt = dap[first_diff_key].float()
        if bt.numel() == dt.numel():
            dt = dt.reshape(bt.shape)
        diff = (bt - dt).abs()

        # Where are the diffs?
        nonzero = (diff > 0).sum().item()
        total = diff.numel()
        print(f"  Non-zero diffs: {nonzero}/{total} ({100*nonzero/total:.4f}%)")

        # Top-10 largest diffs with indices
        flat_diff = diff.flatten()
        top_vals, top_idx = flat_diff.topk(min(10, flat_diff.numel()))
        print(f"  Top diffs:")
        for v, idx in zip(top_vals, top_idx):
            multi_idx = []
            remaining_idx = idx.item()
            for dim in reversed(bt.shape):
                multi_idx.append(remaining_idx % dim)
                remaining_idx //= dim
            multi_idx.reverse()
            bv = bt.flatten()[idx].item()
            dv = dt.flatten()[idx].item()
            print(f"    idx={multi_idx}  base={bv:.6f} dap={dv:.6f} diff={v:.6f}")
    else:
        print(f"  ✅ ALL CHECKPOINTS ARE BIT-IDENTICAL!")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
