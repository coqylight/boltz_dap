#!/usr/bin/env python
"""Compare sub-op and upstream template checkpoints between baseline and DAP."""
import sys
import os
import torch

def compare_checkpoints(base, dap, title, labels):
    print(f"\n{'='*90}")
    print(title)
    print(f"{'='*90}")
    print(f"{'Label':<25s} | {'z_diff_mean':>12s} {'z_diff_max':>12s} {'z_cos':>10s}")
    print("-" * 90)

    for label in labels:
        if label not in base or label not in dap:
            print(f"{label:<25s} | MISSING")
            continue
        zb = base[label].float()
        zd = dap[label].float()
        
        # Handle different batch dims (B*T vs B*T for upstream)
        # Ensure spatial dims match
        N = min(zb.shape[-3], zd.shape[-3])
        M = min(zb.shape[-2], zd.shape[-2])
        zb = zb[..., :N, :M, :]
        zd = zd[..., :N, :M, :]
        
        diff = (zb - zd).abs()
        diff_mean = diff.mean().item()
        diff_max = diff.max().item()
        cos = torch.nn.functional.cosine_similarity(zb.flatten(), zd.flatten(), dim=0).item()

        if diff_mean == 0:
            status = "✅ EXACT"
        elif diff_mean < 0.01:
            status = "✅ TINY"
        elif diff_mean < 0.1:
            status = "⚠️ SMALL"
        elif diff_mean < 1.0:
            status = "⚠️ MEDIUM"
        else:
            status = "❌ LARGE"

        print(f"{label:<25s} | {diff_mean:12.6f} {diff_max:12.4f} {cos:10.6f}  {status}")
    print(f"{'='*90}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_subops.py <baseline_dir_or_file> <dap_dir_or_file>")
        sys.exit(1)
    
    base_path, dap_path = sys.argv[1], sys.argv[2]
    
    # Auto-detect: if paths are directories, look for checkpoint files
    def resolve(p, filename):
        if os.path.isdir(p):
            return os.path.join(p, filename)
        return p
    
    # Compare upstream template checkpoints if available
    base_tmpl = resolve(base_path, "template_upstream_ckpts.pt")
    dap_tmpl = resolve(dap_path, "template_upstream_ckpts.pt")
    if os.path.exists(base_tmpl) and os.path.exists(dap_tmpl):
        base = torch.load(base_tmpl, map_location="cpu", weights_only=True)
        dap = torch.load(dap_tmpl, map_location="cpu", weights_only=True)
        compare_checkpoints(base, dap, "UPSTREAM TEMPLATE CHECKPOINTS", 
                          ["a_tij_full", "z_proj_out", "v_input"])
    
    # Compare sub-op checkpoints
    base_subop = resolve(base_path, "subop_checkpoints.pt")
    dap_subop = resolve(dap_path, "subop_checkpoints.pt")
    if os.path.exists(base_subop) and os.path.exists(dap_subop):
        base = torch.load(base_subop, map_location="cpu", weights_only=True)
        dap = torch.load(dap_subop, map_location="cpu", weights_only=True)
        compare_checkpoints(base, dap, "PER-SUB-OP CHECKPOINTS (PF Layer 0)",
                          ["input", "after_tri_mul_out", "after_tri_mul_in",
                           "after_tri_att_start", "after_tri_att_end", "after_transition"])


if __name__ == "__main__":
    main()
