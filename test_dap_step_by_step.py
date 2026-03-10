"""
Step-by-step full layer comparison: find EXACTLY which step diverges.

Run: torchrun --nproc_per_node=2 --standalone test_dap_step_by_step.py 2>&1 | tee /tmp/dap_steptest.log
"""

import os
import sys
import torch
import torch.distributed as dist
from dataclasses import asdict
from pathlib import Path

PROJ = "/project/engvimmune/gleeai"
sys.path.insert(0, os.path.join(PROJ, "boltz_dap/boltz_dap_v2"))
sys.path.insert(0, os.path.join(PROJ, "boltz_dap"))
sys.path.insert(0, os.path.join(PROJ, "boltz/src"))

from boltz_distributed.core import init_dap, get_dap_size, get_dap_rank
from boltz_distributed.comm import scatter, gather, row_to_col, col_to_row


def rank_print(*args, **kwargs):
    if get_dap_rank() == 0:
        print(*args, **kwargs, flush=True)


def compare(name, original, dap_result):
    if original.shape != dap_result.shape:
        rank_print(f"  ❌ {name}: SHAPE MISMATCH  orig={list(original.shape)} dap={list(dap_result.shape)}")
        min_shape = [min(a, b) for a, b in zip(original.shape, dap_result.shape)]
        slices = tuple(slice(0, s) for s in min_shape)
        original = original[slices]
        dap_result = dap_result[slices]

    diff = (original.float() - dap_result.float()).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    o_flat = original.float().reshape(-1)
    d_flat = dap_result.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(o_flat.unsqueeze(0), d_flat.unsqueeze(0)).item()

    if max_diff < 1e-4:
        v = "✅ MATCH"
    elif max_diff < 1e-2:
        v = "⚠️  CLOSE"
    else:
        v = "❌ DIVERGE"

    rank_print(f"  {v} {name:30s} mean={mean_diff:.2e} max={max_diff:.2e} cos={cos:.6f}")
    return max_diff


def trim(t, N, dims):
    for d in dims:
        if t.shape[d] > N:
            idx = [slice(None)] * len(t.shape)
            idx[d] = slice(0, N)
            t = t[tuple(idx)]
    return t


def gather_and_trim(z_scat, N):
    """Gather scattered tensor and trim to original N."""
    z_full = gather(z_scat.contiguous(), dim=1)
    return trim(z_full, N, [1, 2])


def test_row_col_roundtrip(N=64):
    """Test that row_to_col → col_to_row is identity."""
    rank_print("=" * 70)
    rank_print("TEST 0: row_to_col → col_to_row round-trip")
    rank_print("=" * 70)

    torch.manual_seed(42)
    z = torch.randn(1, N, N, 64, device="cuda", dtype=torch.float32)
    z_scat = scatter(z, dim=1)  # [B, N/2, N, D]

    z_col = row_to_col(z_scat)  # [B, N, N/2, D]
    z_back = col_to_row(z_col)  # [B, N/2, N, D]

    z_back_full = gather_and_trim(z_back, N)
    compare("row→col→row roundtrip", z, z_back_full)

    # Also test col scatter/gather directly
    z_col_full = gather(z_col.contiguous(), dim=2)
    z_col_full = trim(z_col_full, N, [2])
    compare("row→col gather(dim=2)", z, z_col_full)


def test_step_by_step_noseq(model, N=64):
    """Step-by-step comparison of PairformerNoSeqLayer."""
    from dap_trimul import DAPTriMulOut, DAPTriMulIn
    from dap_tri_att import DAPTriAttStart, DAPTriAttEnd
    from dap_pairformer_noseq import get_dropout_mask

    tmpl = model.template_module
    if hasattr(tmpl, '_orig_mod'):
        tmpl = tmpl._orig_mod
    layer = tmpl.pairformer.layers[0]

    D = layer.tri_mul_out.norm_in.normalized_shape[0]

    rank_print("\n" + "=" * 70)
    rank_print(f"TEST 1: Step-by-step PairformerNoSeqLayer (D={D}, N={N})")
    rank_print("=" * 70)

    torch.manual_seed(123)
    z_init = torch.randn(1, N, N, D, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    # Original path
    z_orig = z_init.clone()
    # DAP path
    z_scat = scatter(z_init, dim=1)
    pm_scat = scatter(pm, dim=1)

    dap_trimul_out = DAPTriMulOut(layer.tri_mul_out)
    dap_trimul_in = DAPTriMulIn(layer.tri_mul_in)
    dap_att_start = DAPTriAttStart(layer.tri_att_start)
    dap_att_end = DAPTriAttEnd(layer.tri_att_end)

    with torch.no_grad():
        # ── Step 1: tri_mul_out ──
        rank_print("\n--- Step 1: z += tri_mul_out(z) ---")
        dropout_orig = get_dropout_mask(layer.dropout, z_orig, layer.training)
        z_orig = z_orig + dropout_orig * layer.tri_mul_out(z_orig, mask=pm, use_kernels=False)

        dropout_dap = get_dropout_mask(layer.dropout, z_scat, layer.training)
        z_scat = z_scat + dropout_dap * dap_trimul_out(z_scat, mask=pm_scat, use_kernels=False)

        compare("after tri_mul_out", z_orig, gather_and_trim(z_scat, N))

        # ── Step 2: tri_mul_in ──
        rank_print("\n--- Step 2: z += tri_mul_in(z) ---")
        # Original: operates on full z directly
        dropout_orig = get_dropout_mask(layer.dropout, z_orig, layer.training)
        z_orig = z_orig + dropout_orig * layer.tri_mul_in(z_orig, mask=pm, use_kernels=False)

        # DAP: row_to_col → tri_mul_in → col_to_row
        z_col = row_to_col(z_scat)
        pm_col = row_to_col(pm_scat.unsqueeze(-1)).squeeze(-1)
        dropout_dap = get_dropout_mask(layer.dropout, z_col, layer.training)
        z_col = z_col + dropout_dap * dap_trimul_in(z_col, mask=pm_col, use_kernels=False)
        z_scat = col_to_row(z_col)
        if z_scat.shape[2] > N:
            z_scat = z_scat[:, :, :N, :]

        compare("after tri_mul_in", z_orig, gather_and_trim(z_scat, N))

        # Also check: was the row_to_col→col_to_row lossy?
        # Gather z_scat BEFORE tri_mul_in to see if roundtrip was OK
        rank_print("  (sub-check: row_to_col roundtrip on live z)")
        z_scat_check = scatter(z_init, dim=1)
        z_scat_check = z_scat_check + get_dropout_mask(layer.dropout, z_scat_check, layer.training) * dap_trimul_out(z_scat_check, mask=pm_scat, use_kernels=False)
        # z_scat_check is the z after tri_mul_out — should match z_scat before step 2 began
        z_col_check = row_to_col(z_scat_check)
        z_back_check = col_to_row(z_col_check)
        if z_back_check.shape[2] > N:
            z_back_check = z_back_check[:, :, :N, :]
        compare("row→col→row on z_after_trimulout", gather_and_trim(z_scat_check, N), gather_and_trim(z_back_check, N))

        # ── Step 3: tri_att_start ──
        rank_print("\n--- Step 3: z += tri_att_start(z) ---")
        dropout_orig = get_dropout_mask(layer.dropout, z_orig, layer.training)
        z_orig = z_orig + dropout_orig * layer.tri_att_start(z_orig, mask=pm, chunk_size=None, use_kernels=False)

        dropout_dap = get_dropout_mask(layer.dropout, z_scat, layer.training)
        z_scat = z_scat + dropout_dap * dap_att_start(z_scat, mask=pm_scat, chunk_size=None, use_kernels=False)

        compare("after tri_att_start", z_orig, gather_and_trim(z_scat, N))

        # ── Step 4: tri_att_end ──
        rank_print("\n--- Step 4: z += tri_att_end(z) ---")
        dropout_orig = get_dropout_mask(layer.dropout, z_orig, layer.training, columnwise=True)
        z_orig = z_orig + dropout_orig * layer.tri_att_end(z_orig, mask=pm, chunk_size=None, use_kernels=False)

        dropout_dap = get_dropout_mask(layer.dropout, z_scat, layer.training, columnwise=True)
        z_scat = z_scat + dropout_dap * dap_att_end(z_scat, mask=pm_scat, chunk_size=None, use_kernels=False)

        compare("after tri_att_end", z_orig, gather_and_trim(z_scat, N))

        # ── Step 5: transition_z ──
        rank_print("\n--- Step 5: z += transition_z(z) ---")
        z_orig = z_orig + layer.transition_z(z_orig)
        z_scat = z_scat + layer.transition_z(z_scat, chunk_size=128)

        compare("after transition_z", z_orig, gather_and_trim(z_scat, N))

    rank_print("\n--- Final comparison: manual step-by-step vs DAPPairformerNoSeqLayer ---")
    from dap_pairformer_noseq import DAPPairformerNoSeqLayer
    with torch.no_grad():
        z_scat2 = scatter(z_init, dim=1)
        dap_layer = DAPPairformerNoSeqLayer(layer)
        z_dap2 = dap_layer(z_scat2, pm_scat, chunk_size_tri_attn=None, use_kernels=False)
        compare("manual_steps vs DAPLayer", gather_and_trim(z_scat, N), gather_and_trim(z_dap2, N))
        compare("original vs DAPLayer", z_orig, gather_and_trim(z_dap2, N))


def test_step_by_step_main_pf(model, N=64):
    """Step-by-step for the main PairformerLayer (D=128, with seq attn)."""
    from dap_pairformer import DAPPairformerLayer
    from dap_trimul import DAPTriMulOut, DAPTriMulIn
    from dap_tri_att import DAPTriAttStart, DAPTriAttEnd
    from dap_pairformer_noseq import get_dropout_mask

    pf = model.pairformer_module
    if hasattr(pf, '_orig_mod'):
        pf = pf._orig_mod
    layer = pf.layers[0]

    D_z = layer.tri_mul_out.norm_in.normalized_shape[0]
    D_s = layer.attention.c_s

    rank_print("\n" + "=" * 70)
    rank_print(f"TEST 2: Step-by-step main PairformerLayer (D_z={D_z}, D_s={D_s})")
    rank_print("=" * 70)

    torch.manual_seed(456)
    z_init = torch.randn(1, N, N, D_z, device="cuda", dtype=torch.float32)
    s_init = torch.randn(1, N, D_s, device="cuda", dtype=torch.float32)
    mask = torch.ones(1, N, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    z_orig = z_init.clone()
    s_orig = s_init.clone()
    z_scat = scatter(z_init, dim=1)
    pm_scat = scatter(pm, dim=1)
    s_dap = s_init.clone()

    dap_trimul_out = DAPTriMulOut(layer.tri_mul_out)
    dap_trimul_in = DAPTriMulIn(layer.tri_mul_in)
    dap_att_start = DAPTriAttStart(layer.tri_att_start)
    dap_att_end = DAPTriAttEnd(layer.tri_att_end)

    with torch.no_grad():
        # Step 1: tri_mul_out
        rank_print("\n--- Step 1: z += tri_mul_out(z) ---")
        dropout_o = get_dropout_mask(layer.dropout, z_orig, layer.training)
        z_orig = z_orig + dropout_o * layer.tri_mul_out(z_orig, mask=pm, use_kernels=False)
        dropout_d = get_dropout_mask(layer.dropout, z_scat, layer.training)
        z_scat = z_scat + dropout_d * dap_trimul_out(z_scat, mask=pm_scat, use_kernels=False)
        compare("after tri_mul_out", z_orig, gather_and_trim(z_scat, N))

        # Step 2: tri_mul_in
        rank_print("\n--- Step 2: z += tri_mul_in(z) ---")
        dropout_o = get_dropout_mask(layer.dropout, z_orig, layer.training)
        z_orig = z_orig + dropout_o * layer.tri_mul_in(z_orig, mask=pm, use_kernels=False)
        z_col = row_to_col(z_scat)
        pm_col = row_to_col(pm_scat.unsqueeze(-1)).squeeze(-1)
        dropout_d = get_dropout_mask(layer.dropout, z_col, layer.training)
        z_col = z_col + dropout_d * dap_trimul_in(z_col, mask=pm_col, use_kernels=False)
        z_scat = col_to_row(z_col)
        if z_scat.shape[2] > N:
            z_scat = z_scat[:, :, :N, :]
        compare("after tri_mul_in", z_orig, gather_and_trim(z_scat, N))

        # Step 3: tri_att_start
        rank_print("\n--- Step 3: z += tri_att_start(z) ---")
        dropout_o = get_dropout_mask(layer.dropout, z_orig, layer.training)
        z_orig = z_orig + dropout_o * layer.tri_att_start(z_orig, mask=pm, chunk_size=None, use_kernels=False)
        dropout_d = get_dropout_mask(layer.dropout, z_scat, layer.training)
        z_scat = z_scat + dropout_d * dap_att_start(z_scat, mask=pm_scat, chunk_size=None, use_kernels=False)
        compare("after tri_att_start", z_orig, gather_and_trim(z_scat, N))

        # Step 4: tri_att_end
        rank_print("\n--- Step 4: z += tri_att_end(z) ---")
        dropout_o = get_dropout_mask(layer.dropout, z_orig, layer.training, columnwise=True)
        z_orig = z_orig + dropout_o * layer.tri_att_end(z_orig, mask=pm, chunk_size=None, use_kernels=False)
        dropout_d = get_dropout_mask(layer.dropout, z_scat, layer.training, columnwise=True)
        z_scat = z_scat + dropout_d * dap_att_end(z_scat, mask=pm_scat, chunk_size=None, use_kernels=False)
        compare("after tri_att_end", z_orig, gather_and_trim(z_scat, N))

        # Step 5: transition_z
        rank_print("\n--- Step 5: z += transition_z(z) ---")
        z_orig = z_orig + layer.transition_z(z_orig)
        z_scat = z_scat + layer.transition_z(z_scat, chunk_size=128)
        compare("after transition_z", z_orig, gather_and_trim(z_scat, N))

        # Step 6: sequence attention (DAP version)
        rank_print("\n--- Step 6: s += seq_attention(s, z) ---")
        s_normed_o = layer.pre_norm_s(s_orig)
        s_orig = s_orig + layer.attention(s=s_normed_o, z=z_orig, mask=mask, k_in=s_normed_o)

        # DAP seq attention: compute bias on scattered z, gather
        attn_mod = layer.attention
        pair_bias = attn_mod.proj_z(z_scat)  # on scattered z
        pair_bias = gather(pair_bias.contiguous(), dim=2, original_size=N)
        s_normed_d = layer.pre_norm_s(s_dap)
        B = s_normed_d.shape[0]
        q = attn_mod.proj_q(s_normed_d).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
        k = attn_mod.proj_k(s_normed_d).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
        v = attn_mod.proj_v(s_normed_d).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
        g = attn_mod.proj_g(s_normed_d).sigmoid()
        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (attn_mod.head_dim ** 0.5) + pair_bias.float()
            attn = attn + (1 - mask[:, None, None].float()) * -attn_mod.inf
            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, attn_mod.c_s)
        s_dap = s_dap + attn_mod.proj_o(g * o)
        compare("after seq_attn (s)", s_orig, s_dap)


def main():
    init_dap()
    rank_print("=" * 70)
    rank_print(f"STEP-BY-STEP DAP COMPARISON (dap_size={get_dap_size()})")
    rank_print("=" * 70)

    rank_print("\nLoading model...")
    from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2
    cache = Path("~/.boltz").expanduser()
    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt", strict=True,
        predict_args={"recycling_steps": 3, "sampling_steps": 200,
                      "diffusion_samples": 1, "max_parallel_samples": 1,
                      "write_confidence_summary": True, "write_full_pae": False,
                      "write_full_pde": False},
        map_location=f"cuda:{get_dap_rank()}",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False, use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    rank_print("Model loaded.\n")

    # Suppress DAP profiling noise
    import dap_trimul
    dap_trimul.DIAG = False

    test_row_col_roundtrip()
    test_step_by_step_noseq(model)
    test_step_by_step_main_pf(model)

    rank_print("\n" + "=" * 70)
    rank_print("ALL STEP-BY-STEP TESTS COMPLETE")
    rank_print("=" * 70)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
