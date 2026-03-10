"""
Layer-level comparison: Original Boltz vs DAP implementation.

Run with torchrun on 2 GPUs:
    torchrun --nproc_per_node=2 --standalone test_dap_layer_comparison.py

Tests each sub-operation to find exactly which one diverges.
Auto-detects dimensions from model weights.
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
    """Compare two tensors and print diff stats."""
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
    """Trim tensor to N on specified dims."""
    for d in dims:
        if t.shape[d] > N:
            idx = [slice(None)] * len(t.shape)
            idx[d] = slice(0, N)
            t = t[tuple(idx)]
    return t


def get_dim(layer, attr_path):
    """Get the normalized_shape from a LayerNorm to auto-detect dimension."""
    obj = layer
    for a in attr_path.split('.'):
        obj = getattr(obj, a)
    return obj.normalized_shape[0]


def test_pairformer_noseq(model):
    """Test PairformerNoSeqLayer sub-operations (template pairformer)."""
    from dap_pairformer_noseq import DAPPairformerNoSeqLayer
    from dap_trimul import DAPTriMulOut, DAPTriMulIn
    from dap_tri_att import DAPTriAttStart, DAPTriAttEnd

    tmpl = model.template_module
    if hasattr(tmpl, '_orig_mod'):
        tmpl = tmpl._orig_mod
    layer = tmpl.pairformer.layers[0]

    # Auto-detect dim from the layer's LayerNorm
    D = get_dim(layer, 'tri_mul_out.norm_in')
    N = 64

    rank_print("=" * 70)
    rank_print(f"TEST 1: PairformerNoSeqLayer (template, D={D}, N={N})")
    rank_print("=" * 70)

    torch.manual_seed(123)
    z = torch.randn(1, N, N, D, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    z_scat = scatter(z, dim=1)
    pm_scat = scatter(pm, dim=1)

    with torch.no_grad():
        # 1. tri_mul_out
        rank_print("\n--- tri_mul_out ---")
        orig = layer.tri_mul_out(z, mask=pm, use_kernels=False)
        dap = DAPTriMulOut(layer.tri_mul_out)(z_scat, mask=pm_scat, use_kernels=False)
        dap_g = trim(gather(dap.contiguous(), dim=1), N, [1])
        compare("tri_mul_out", orig, dap_g)

        # 2. tri_mul_in
        rank_print("\n--- tri_mul_in ---")
        orig = layer.tri_mul_in(z, mask=pm, use_kernels=False)
        z_col = row_to_col(z_scat)
        pm_col = row_to_col(pm_scat.unsqueeze(-1)).squeeze(-1)
        dap = DAPTriMulIn(layer.tri_mul_in)(z_col, mask=pm_col, use_kernels=False)
        dap_row = col_to_row(dap)
        dap_g = trim(gather(dap_row.contiguous(), dim=1), N, [1, 2])
        compare("tri_mul_in", orig, dap_g)

        # 3. tri_att_start
        rank_print("\n--- tri_att_start ---")
        orig = layer.tri_att_start(z, mask=pm, chunk_size=None, use_kernels=False)
        dap = DAPTriAttStart(layer.tri_att_start)(z_scat, mask=pm_scat, chunk_size=None, use_kernels=False)
        dap_g = trim(gather(dap.contiguous(), dim=1), N, [1])
        compare("tri_att_start", orig, dap_g)

        # 4. tri_att_end
        rank_print("\n--- tri_att_end ---")
        orig = layer.tri_att_end(z, mask=pm, chunk_size=None, use_kernels=False)
        dap = DAPTriAttEnd(layer.tri_att_end)(z_scat, mask=pm_scat, chunk_size=None, use_kernels=False)
        dap_g = trim(gather(dap.contiguous(), dim=1), N, [1])
        compare("tri_att_end", orig, dap_g)

        # 5. transition_z
        rank_print("\n--- transition_z ---")
        orig = layer.transition_z(z)
        dap_c = layer.transition_z(z_scat, chunk_size=128)
        dap_g = trim(gather(dap_c.contiguous(), dim=1), N, [1])
        compare("transition_z(chunk=128)", orig, dap_g)

        dap2 = layer.transition_z(z_scat)
        dap_g2 = trim(gather(dap2.contiguous(), dim=1), N, [1])
        compare("transition_z(no_chunk)", orig, dap_g2)

        # 6. Full layer
        rank_print("\n--- Full PairformerNoSeqLayer ---")
        orig = layer(z, pm, chunk_size_tri_attn=None, use_kernels=False)
        dap_layer = DAPPairformerNoSeqLayer(layer)
        dap = dap_layer(z_scat, pm_scat, chunk_size_tri_attn=None, use_kernels=False)
        dap_g = trim(gather(dap.contiguous(), dim=1), N, [1, 2])
        compare("full_noseq_layer", orig, dap_g)


def test_pairformer_with_seq(model):
    """Test PairformerLayer with sequence attention (main pairformer)."""
    from dap_pairformer import DAPPairformerLayer

    pf = model.pairformer_module
    if hasattr(pf, '_orig_mod'):
        pf = pf._orig_mod
    layer = pf.layers[0]

    D_z = get_dim(layer, 'tri_mul_out.norm_in')
    D_s = layer.attention.c_s
    N = 64

    rank_print("\n" + "=" * 70)
    rank_print(f"TEST 2: PairformerLayer (main, D_z={D_z}, D_s={D_s}, N={N})")
    rank_print("=" * 70)

    torch.manual_seed(456)
    z = torch.randn(1, N, N, D_z, device="cuda", dtype=torch.float32)
    s = torch.randn(1, N, D_s, device="cuda", dtype=torch.float32)
    mask = torch.ones(1, N, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        # Test proj_z bias
        rank_print("\n--- proj_z bias ---")
        orig_bias = layer.attention.proj_z(z.float())  # [B, H, N, N]
        z_scat = scatter(z, dim=1)
        dap_bias = layer.attention.proj_z(z_scat.float())  # [B, H, N/dap, N]
        dap_bias_g = trim(gather(dap_bias.contiguous(), dim=2), N, [2])
        compare("proj_z_bias", orig_bias, dap_bias_g)

        # Test sequence attention
        rank_print("\n--- Sequence attention ---")
        s_normed = layer.pre_norm_s(s.float())
        orig_attn = layer.attention(s=s_normed, z=z.float(), mask=mask.float(), k_in=s_normed)

        attn_mod = layer.attention
        pair_bias = attn_mod.proj_z(z_scat.float())
        pair_bias = gather(pair_bias.contiguous(), dim=2, original_size=N)

        B = s_normed.shape[0]
        q = attn_mod.proj_q(s_normed).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
        k = attn_mod.proj_k(s_normed).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
        v = attn_mod.proj_v(s_normed).view(B, -1, attn_mod.num_heads, attn_mod.head_dim)
        g = attn_mod.proj_g(s_normed).sigmoid()

        with torch.autocast("cuda", enabled=False):
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (attn_mod.head_dim ** 0.5) + pair_bias.float()
            attn = attn + (1 - mask[:, None, None].float()) * -attn_mod.inf
            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, attn_mod.c_s)
        dap_attn = attn_mod.proj_o(g * o)
        compare("seq_attention", orig_attn, dap_attn)

        # Full layer
        rank_print("\n--- Full PairformerLayer ---")
        s_orig, z_orig = layer(s.clone(), z.clone(), mask, pm, chunk_size_tri_attn=None, use_kernels=False)

        z_scat = scatter(z, dim=1)
        pm_scat = scatter(pm, dim=1)
        dap_layer = DAPPairformerLayer(layer)
        s_dap, z_dap_scat = dap_layer(s.clone(), z_scat, mask, pm_scat, chunk_size_tri_attn=None, use_kernels=False)
        z_dap = trim(gather(z_dap_scat.contiguous(), dim=1), N, [1, 2])

        compare("s (sequence)", s_orig, s_dap)
        compare("z (pair)", z_orig, z_dap)


def test_cumulative(model, num_layers=8):
    """Measure divergence accumulation over multiple layers."""
    from dap_pairformer_noseq import DAPPairformerNoSeqLayer

    tmpl = model.template_module
    if hasattr(tmpl, '_orig_mod'):
        tmpl = tmpl._orig_mod
    layers = tmpl.pairformer.layers

    D = get_dim(layers[0], 'tri_mul_out.norm_in')
    N = 64

    rank_print("\n" + "=" * 70)
    rank_print(f"TEST 3: Cumulative Divergence ({num_layers} layers, D={D})")
    rank_print("=" * 70)

    torch.manual_seed(789)
    z = torch.randn(1, N, N, D, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    z_orig = z.clone()
    z_scat = scatter(z, dim=1)
    pm_scat = scatter(pm, dim=1)

    n = min(num_layers, len(layers))
    with torch.no_grad():
        for i in range(n):
            z_orig = layers[i](z_orig, pm, chunk_size_tri_attn=None, use_kernels=False)
            dap_layer = DAPPairformerNoSeqLayer(layers[i])
            z_scat = dap_layer(z_scat, pm_scat, chunk_size_tri_attn=None, use_kernels=False)
            z_dap_full = trim(gather(z_scat.contiguous(), dim=1), N, [1, 2])
            compare(f"layer_{i}", z_orig, z_dap_full)


def test_opm(model):
    """Test outer product mean."""
    from dap_msa import _opm_scattered

    msa_mod = model.msa_module
    if hasattr(msa_mod, '_orig_mod'):
        msa_mod = msa_mod._orig_mod

    opm = msa_mod.layers[0].outer_product_mean
    N = 64
    S = 16

    rank_print("\n" + "=" * 70)
    rank_print("TEST 4: Outer Product Mean")
    rank_print("=" * 70)

    torch.manual_seed(321)
    c_in = opm.proj_a.in_features
    m = torch.randn(1, S, N, c_in, device="cuda", dtype=torch.float32)
    msa_mask = torch.ones(1, S, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        rank_print("\n--- outer_product_mean ---")
        orig = opm(m, msa_mask, chunk_size=None)
        dap = _opm_scattered(opm, m, msa_mask, chunk_size=None)
        dap_g = trim(gather(dap.contiguous(), dim=1), N, [1])
        compare("outer_product_mean", orig, dap_g)


def test_pwa(model):
    """Test pair-weighted averaging."""
    from dap_msa import _pwa_with_bias

    msa_mod = model.msa_module
    if hasattr(msa_mod, '_orig_mod'):
        msa_mod = msa_mod._orig_mod

    pwa = msa_mod.layers[0].pair_weighted_averaging
    D_z = pwa.proj_z[0].normalized_shape[0]  # LayerNorm inside proj_z Sequential
    N = 64
    S = 16

    rank_print("\n" + "=" * 70)
    rank_print(f"TEST 5: Pair Weighted Averaging (D_z={D_z})")
    rank_print("=" * 70)

    torch.manual_seed(555)
    c_m = pwa.proj_m.in_features
    m = torch.randn(1, S, N, c_m, device="cuda", dtype=torch.float32)
    z = torch.randn(1, N, N, D_z, device="cuda", dtype=torch.float32)
    token_mask = torch.ones(1, N, N, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        rank_print("\n--- pair_weighted_averaging ---")
        # Original
        orig = pwa(m, z, token_mask, chunk_heads=False)

        # DAP: compute bias on scattered z, gather, then run with bias
        z_scat = scatter(z, dim=1)
        z_normed_scat = pwa.norm_z(z_scat)
        b_scat = pwa.proj_z(z_normed_scat)  # [B, N/dap, N, H]
        b_full = gather(b_scat.contiguous(), dim=1, original_size=N)
        m_normed = pwa.norm_m(m)
        dap = _pwa_with_bias(pwa, m_normed, b_full, token_mask, chunk_heads=False)

        compare("pwa", orig, dap)


def main():
    init_dap()
    dap_rank = get_dap_rank()
    dap_size = get_dap_size()

    rank_print("=" * 70)
    rank_print(f"DAP LAYER-LEVEL COMPARISON TEST (dap_size={dap_size})")
    rank_print("=" * 70)

    rank_print("\nLoading model weights...")
    from boltz.main import Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams
    from boltz.model.models.boltz2 import Boltz2

    cache = Path("~/.boltz").expanduser()
    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt",
        strict=True,
        predict_args={
            "recycling_steps": 3, "sampling_steps": 200,
            "diffusion_samples": 1, "max_parallel_samples": 1,
            "write_confidence_summary": True, "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location=f"cuda:{dap_rank}",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model.eval()
    rank_print("Model loaded.\n")

    test_pairformer_noseq(model)
    test_pairformer_with_seq(model)
    test_cumulative(model)
    test_opm(model)
    test_pwa(model)

    rank_print("\n" + "=" * 70)
    rank_print("ALL TESTS COMPLETE")
    rank_print("=" * 70)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
