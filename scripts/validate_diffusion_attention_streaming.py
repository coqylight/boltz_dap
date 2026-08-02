from __future__ import annotations

from dataclasses import asdict
import gc
import os
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "boltz" / "src"))

from boltz.model.layers.attentionv2 import AttentionPairBias
from boltz.model.modules.transformersv2 import DiffusionTransformer


def mib(value: int) -> float:
    return value / 1024**2


def validate_transformer_parity() -> None:
    torch.manual_seed(982891216)
    device = torch.device("cuda")
    tokens = 257
    channels = 64
    heads = 8
    depth = 2
    transformer = DiffusionTransformer(
        depth=depth,
        heads=heads,
        dim=channels,
        dim_single_cond=channels,
    ).to(device).eval()
    sequence = torch.randn(1, tokens, channels, device=device)
    conditioning = torch.randn_like(sequence)
    mask = torch.ones(1, tokens, device=device)
    mask[:, -5:] = 0
    bias = torch.randn(1, tokens, tokens, depth * heads, dtype=torch.bfloat16)

    with torch.no_grad():
        os.environ["BOLTZ_DIFFUSION_ATTN_Q_CHUNK"] = "0"
        expected = transformer(
            sequence.clone(),
            conditioning,
            bias=bias.to(device),
            mask=mask,
        )
        os.environ["BOLTZ_DIFFUSION_ATTN_Q_CHUNK"] = "31"
        streamed = transformer(
            sequence.clone(),
            conditioning,
            bias=list(bias.chunk(depth, dim=-1)),
            mask=mask,
        )

    difference = (streamed - expected).abs()
    print(
        "transformer parity: "
        f"mean={difference.mean().item():.6e} "
        f"max={difference.max().item():.6e}",
        flush=True,
    )
    torch.testing.assert_close(streamed, expected, rtol=2e-5, atol=2e-5)


def validate_checkpoint_transformer_parity() -> None:
    from boltz.main import (
        Boltz2DiffusionParams,
        BoltzSteeringParams,
        MSAModuleArgs,
        PairformerArgsV2,
    )
    from boltz.model.models.boltz2 import Boltz2

    device = torch.device("cuda")
    checkpoint = Path("~/.boltz/boltz2_conf.ckpt").expanduser()
    model = Boltz2.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args={
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        },
        map_location="cpu",
        diffusion_process_args=asdict(Boltz2DiffusionParams()),
        ema=False,
        use_kernels=False,
        pairformer_args=asdict(PairformerArgsV2()),
        msa_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    transformer = model.structure_module.score_model.token_transformer.to(device).eval()
    depth = len(transformer.layers)
    attention = transformer.layers[0].pair_bias_attn
    channels = attention.c_s
    heads = attention.num_heads
    tokens = 73
    torch.manual_seed(982891216)
    sequence = torch.randn(1, tokens, channels, device=device)
    conditioning = torch.randn_like(sequence)
    mask = torch.ones(1, tokens, device=device)
    mask[:, -5:] = 0
    bias = torch.randn(1, tokens, tokens, depth * heads, dtype=torch.bfloat16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        os.environ["BOLTZ_DIFFUSION_ATTN_Q_CHUNK"] = "0"
        expected = transformer(
            sequence.clone(),
            conditioning,
            bias=bias.to(device),
            mask=mask,
        )
        os.environ["BOLTZ_DIFFUSION_ATTN_Q_CHUNK"] = "31"
        streamed = transformer(
            sequence.clone(),
            conditioning,
            bias=list(bias.chunk(depth, dim=-1)),
            mask=mask,
        )

    difference = (streamed - expected).abs()
    print(
        "checkpoint transformer parity: "
        f"layers={depth} channels={channels} heads={heads} "
        f"mean={difference.mean().item():.6e} "
        f"max={difference.max().item():.6e}",
        flush=True,
    )
    torch.testing.assert_close(streamed, expected, rtol=2e-5, atol=2e-5)
    transformer.cpu()
    del model, transformer, sequence, conditioning, mask, bias, expected, streamed
    gc.collect()
    torch.cuda.empty_cache()


def validate_exact_shape_memory(query_chunk: int) -> None:
    device = torch.device("cuda")
    tokens = int(os.environ.get("DIFFUSION_ATTN_VALIDATION_TOKENS", "9342"))
    entry_mib = int(os.environ.get("DIFFUSION_ATTN_VALIDATION_ENTRY_MIB", "0"))
    channels = 768
    heads = 16
    attention = AttentionPairBias(
        c_s=channels,
        num_heads=heads,
        compute_pair_bias=False,
    ).to(device=device, dtype=torch.bfloat16).eval()
    sequence = torch.zeros(1, tokens, channels, device=device, dtype=torch.bfloat16)
    mask = torch.ones(1, tokens, device=device, dtype=torch.bfloat16)
    bias = torch.zeros(1, tokens, tokens, heads, dtype=torch.bfloat16)

    os.environ["BOLTZ_DIFFUSION_ATTN_Q_CHUNK"] = str(query_chunk)
    torch.cuda.empty_cache()
    target_entry = entry_mib * 1024**2
    floor_reservations = []
    while torch.cuda.memory_allocated() < target_entry:
        reservation_bytes = min(
            4 * 1024**3,
            target_entry - torch.cuda.memory_allocated(),
        )
        floor_reservations.append(
            torch.empty(reservation_bytes, dtype=torch.uint8, device=device)
        )
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        output = attention(sequence, bias, mask, sequence)
    end.record()
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - before
    full_score_bytes = tokens * tokens * heads * torch.float32.itemsize
    one_tile_bytes = query_chunk * tokens * heads * torch.float32.itemsize
    print(
        "exact-shape memory: "
        f"N={tokens} q_chunk={query_chunk} "
        f"entry={mib(before):.1f} MiB "
        f"peak_extra={mib(peak_extra):.1f} MiB "
        f"full_score={mib(full_score_bytes):.1f} MiB "
        f"one_tile={mib(one_tile_bytes):.1f} MiB "
        f"elapsed={start.elapsed_time(end) / 1000:.3f}s",
        flush=True,
    )
    if not torch.isfinite(output).all():
        raise AssertionError("streamed attention returned non-finite output")
    if peak_extra >= full_score_bytes:
        raise AssertionError(
            "streamed attention allocated at least one full fp32 score matrix: "
            f"peak_extra={peak_extra}, full_score_bytes={full_score_bytes}"
        )
    if bias.device.type != "cpu":
        raise AssertionError("exact-shape pair bias did not remain CPU-backed")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    query_chunk = int(os.environ.get("BOLTZ_DIFFUSION_ATTN_Q_CHUNK", "512"))
    print(f"device={torch.cuda.get_device_name()} vram={mib(torch.cuda.get_device_properties(0).total_memory):.1f} MiB")
    validate_transformer_parity()
    validate_checkpoint_transformer_parity()
    validate_exact_shape_memory(query_chunk)
    print("diffusion_attention_streaming_validation=PASS", flush=True)


if __name__ == "__main__":
    main()