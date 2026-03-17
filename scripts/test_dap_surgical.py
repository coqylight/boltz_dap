"""
SURGICAL test: instrument DAPPairformerNoSeqLayer to find exactly where it diverges.

Run: torchrun --nproc_per_node=2 --standalone test_dap_surgical.py 2>&1 | tee /project/engvimmune/gleeai/boltz_dap/surgical_output.log
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

def rank_print(*a, **k):
    if get_dap_rank() == 0:
        print(*a, **k, flush=True)

def compare(name, a, b):
    if a.shape != b.shape:
        rank_print(f"  ❌ {name}: SHAPE {list(a.shape)} vs {list(b.shape)}")
        return
    diff = (a.float() - b.float()).abs()
    m = diff.mean().item(); mx = diff.max().item()
    v = "✅" if mx < 1e-4 else ("⚠️ " if mx < 1e-2 else "❌")
    rank_print(f"  {v} {name:40s} mean={m:.2e} max={mx:.2e}")
    return mx

def trim(t, N, dims):
    for d in dims:
        if t.shape[d] > N:
            idx = [slice(None)] * len(t.shape)
            idx[d] = slice(0, N)
            t = t[tuple(idx)]
    return t

def g(z_scat, N):
    return trim(gather(z_scat.contiguous(), dim=1), N, [1, 2])


def main():
    init_dap()
    import dap_trimul; dap_trimul.DIAG = False

    rank_print("Loading model...")
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

    tmpl = model.template_module
    if hasattr(tmpl, '_orig_mod'):
        tmpl = tmpl._orig_mod
    layer = tmpl.pairformer.layers[0]

    from dap_pairformer_noseq import DAPPairformerNoSeqLayer, get_dropout_mask
    from dap_trimul import DAPTriMulOut, DAPTriMulIn
    from dap_tri_att import DAPTriAttStart, DAPTriAttEnd

    D = layer.tri_mul_out.norm_in.normalized_shape[0]
    N = 64

    torch.manual_seed(123)
    z_init = torch.randn(1, N, N, D, device="cuda", dtype=torch.float32)
    pm = torch.ones(1, N, N, device="cuda", dtype=torch.float32)
    z_scat = scatter(z_init, dim=1)
    pm_scat = scatter(pm, dim=1)

    rank_print("=" * 70)
    rank_print("SURGICAL TEST: Instrument DAPPairformerNoSeqLayer internals")
    rank_print("=" * 70)

    # ── Create the DAP layer ──
    dap_layer = DAPPairformerNoSeqLayer(layer)

    # ── Monkey-patch forward to capture intermediates ──
    intermediates = {}

    _orig_forward = dap_layer.forward
    def instrumented_forward(z, pair_mask, **kwargs):
        original_N = z.shape[2]
        dap_rank = get_dap_rank()

        intermediates['input_z'] = z.clone()

        # 1. TriMulOut
        dropout = get_dropout_mask(dap_layer.dropout, z, dap_layer.training)
        intermediates['dropout_1'] = dropout if isinstance(dropout, float) else dropout.clone()
        tri_out = dap_layer.tri_mul_out(z, mask=pair_mask, use_kernels=False)
        intermediates['tri_mul_out_output'] = tri_out.clone()
        z = z + dropout * tri_out
        intermediates['after_step1'] = z.clone()

        # 2. TriMulIn
        z_col = row_to_col(z)
        intermediates['z_col_before_trimin'] = z_col.clone()
        pair_mask_col = row_to_col(pair_mask.unsqueeze(-1)).squeeze(-1)
        dropout = get_dropout_mask(dap_layer.dropout, z_col, dap_layer.training)
        intermediates['dropout_2'] = dropout if isinstance(dropout, float) else dropout.clone()
        tri_in_out = dap_layer.tri_mul_in(z_col, mask=pair_mask_col, use_kernels=False)
        intermediates['tri_mul_in_output'] = tri_in_out.clone()
        z_col = z_col + dropout * tri_in_out
        z = col_to_row(z_col)
        del z_col
        if z.shape[2] > original_N:
            z = z[:, :, :original_N, :]
        intermediates['after_step2'] = z.clone()

        # 3. TriAttStart
        dropout = get_dropout_mask(dap_layer.dropout, z, dap_layer.training)
        intermediates['dropout_3'] = dropout if isinstance(dropout, float) else dropout.clone()
        tri_att_s = dap_layer.tri_att_start(z, mask=pair_mask, chunk_size=None, use_kernels=False)
        intermediates['tri_att_start_output'] = tri_att_s.clone()
        z = z + dropout * tri_att_s
        intermediates['after_step3'] = z.clone()

        # 4. TriAttEnd
        dropout = get_dropout_mask(dap_layer.dropout, z, dap_layer.training, columnwise=True)
        intermediates['dropout_4'] = dropout if isinstance(dropout, float) else dropout.clone()
        tri_att_e = dap_layer.tri_att_end(z, mask=pair_mask, chunk_size=None, use_kernels=False)
        intermediates['tri_att_end_output'] = tri_att_e.clone()
        z = z + dropout * tri_att_e
        intermediates['after_step4'] = z.clone()

        # 5. Transition
        trans = dap_layer.transition_z(z, chunk_size=128)
        intermediates['transition_output'] = trans.clone()
        z = z + trans
        intermediates['after_step5'] = z.clone()

        return z

    # ── Run the INSTRUMENTED forward ──
    with torch.no_grad():
        z_dap = instrumented_forward(z_scat.clone(), pm_scat)

    # ── Now run the MANUAL step-by-step (known working) ──
    manual = {}
    z_m = z_scat.clone()
    with torch.no_grad():
        # 1
        d1 = get_dropout_mask(layer.dropout, z_m, layer.training)
        manual['dropout_1'] = d1
        out1 = DAPTriMulOut(layer.tri_mul_out)(z_m, mask=pm_scat, use_kernels=False)
        manual['tri_mul_out_output'] = out1.clone()
        z_m = z_m + d1 * out1
        manual['after_step1'] = z_m.clone()

        # 2
        z_col = row_to_col(z_m)
        pm_col = row_to_col(pm_scat.unsqueeze(-1)).squeeze(-1)
        d2 = get_dropout_mask(layer.dropout, z_col, layer.training)
        manual['dropout_2'] = d2
        out2 = DAPTriMulIn(layer.tri_mul_in)(z_col, mask=pm_col, use_kernels=False)
        manual['tri_mul_in_output'] = out2.clone()
        z_col = z_col + d2 * out2
        z_m = col_to_row(z_col)
        if z_m.shape[2] > N:
            z_m = z_m[:, :, :N, :]
        manual['after_step2'] = z_m.clone()

        # 3
        d3 = get_dropout_mask(layer.dropout, z_m, layer.training)
        manual['dropout_3'] = d3
        out3 = DAPTriAttStart(layer.tri_att_start)(z_m, mask=pm_scat, chunk_size=None, use_kernels=False)
        manual['tri_att_start_output'] = out3.clone()
        z_m = z_m + d3 * out3
        manual['after_step3'] = z_m.clone()

        # 4
        d4 = get_dropout_mask(layer.dropout, z_m, layer.training, columnwise=True)
        manual['dropout_4'] = d4
        out4 = DAPTriAttEnd(layer.tri_att_end)(z_m, mask=pm_scat, chunk_size=None, use_kernels=False)
        manual['tri_att_end_output'] = out4.clone()
        z_m = z_m + d4 * out4
        manual['after_step4'] = z_m.clone()

        # 5
        out5 = layer.transition_z(z_m, chunk_size=128)
        manual['transition_output'] = out5.clone()
        z_m = z_m + out5
        manual['after_step5'] = z_m.clone()

    # ── Compare step by step ──
    rank_print("\n--- Comparing instrumented DAPLayer vs manual steps ---\n")
    for key in ['after_step1', 'tri_mul_out_output',
                'after_step2', 'tri_mul_in_output',
                'after_step3', 'tri_att_start_output',
                'after_step4', 'tri_att_end_output',
                'after_step5', 'transition_output']:
        a = intermediates[key]
        b = manual[key]
        # Both are scattered, compare directly (no gather needed)
        compare(f"scat: {key}", a, b)

    # Also compare gathered versions
    rank_print("\n--- Gathered comparisons ---\n")
    for key in ['after_step1', 'after_step2', 'after_step3', 'after_step4', 'after_step5']:
        a = g(intermediates[key], N)
        b = g(manual[key], N)
        compare(f"full: {key}", a, b)

    # ── Check: are the DAPTriMulOut objects using the same underlying module? ──
    rank_print("\n--- Module identity checks ---")
    rank_print(f"  dap_layer.tri_mul_out.inner is layer.tri_mul_out: "
               f"{dap_layer.tri_mul_out.inner is layer.tri_mul_out}")
    rank_print(f"  dap_layer.tri_mul_in.inner is layer.tri_mul_in: "
               f"{dap_layer.tri_mul_in.inner is layer.tri_mul_in}")
    rank_print(f"  dap_layer.tri_att_start.inner is layer.tri_att_start: "
               f"{dap_layer.tri_att_start.inner is layer.tri_att_start}")
    rank_print(f"  dap_layer.tri_att_end.inner is layer.tri_att_end: "
               f"{dap_layer.tri_att_end.inner is layer.tri_att_end}")
    rank_print(f"  dap_layer.transition_z is layer.transition_z: "
               f"{dap_layer.transition_z is layer.transition_z}")

    # ── Check: dap_layer.training vs layer.training ──
    rank_print(f"\n  dap_layer.training: {dap_layer.training}")
    rank_print(f"  layer.training: {layer.training}")
    rank_print(f"  dap_layer.dropout: {dap_layer.dropout}")
    rank_print(f"  layer.dropout: {layer.dropout}")

    # ── Also run the ORIGINAL forward (not instrumented) to compare ──
    rank_print("\n--- Comparing: original DAPLayer.forward() vs instrumented ---")
    dap_layer2 = DAPPairformerNoSeqLayer(layer)
    with torch.no_grad():
        z_dap2 = dap_layer2(z_scat.clone(), pm_scat, chunk_size_tri_attn=None, use_kernels=False)
    compare("dap_layer vs dap_layer2 (both fresh)", z_dap, z_dap2)
    compare("instrumented vs manual final", g(z_dap, N), g(z_m, N))

    rank_print("\n" + "=" * 70)
    rank_print("DONE")
    rank_print("=" * 70)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
