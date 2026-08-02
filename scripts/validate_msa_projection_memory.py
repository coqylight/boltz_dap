#!/usr/bin/env python
import argparse
import os
import sys

import torch
from torch import nn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "boltz_dap_v2"))
sys.path.insert(0, os.path.join(ROOT, "..", "boltz", "src"))

from boltz.data import const
from dap_trunk import _project_msa_features_chunked


def _features(num_sequences: int, num_tokens: int, device: torch.device):
    msa = torch.arange(
        num_sequences * num_tokens, device=device, dtype=torch.long
    ).reshape(1, num_sequences, num_tokens)
    msa.remainder_(const.num_tokens)
    scalar_shape = (1, num_sequences, num_tokens, 1)
    has_deletion = torch.zeros(scalar_shape, device=device)
    deletion_value = torch.zeros(scalar_shape, device=device)
    is_paired = torch.zeros(scalar_shape, device=device)
    return msa, has_deletion, deletion_value, is_paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-sequences", type=int, default=4224)
    parser.add_argument("--num-tokens", type=int, default=10272)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--sequence-chunk", type=int, default=4)
    parser.add_argument("--entry-floor-mib", type=int, default=64842)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", 0)
    projection = nn.Linear(const.num_tokens + 3, args.channels, bias=False).to(device)
    torch.manual_seed(7)
    nn.init.normal_(projection.weight, std=0.02)

    small = _features(16, 256, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full_features = torch.cat(
            [
                torch.nn.functional.one_hot(small[0], num_classes=const.num_tokens),
                small[1],
                small[2],
                small[3],
            ],
            dim=-1,
        )
        expected = projection(full_features)
        actual = _project_msa_features_chunked(
            projection,
            *small,
            use_paired_feature=True,
            num_tokens=const.num_tokens,
            sequence_chunk=args.sequence_chunk,
        )
    torch.testing.assert_close(actual, expected)
    del small, full_features, expected, actual
    torch.cuda.empty_cache()

    features = _features(args.num_sequences, args.num_tokens, device)
    current_bytes = torch.cuda.memory_allocated(device)
    target_bytes = args.entry_floor_mib * 1024**2
    if current_bytes >= target_bytes:
        raise AssertionError(
            f"validation inputs already exceed entry floor: {current_bytes / 1024**2:.1f} MiB"
        )
    floor_reservation = torch.empty(
        target_bytes - current_bytes, dtype=torch.uint8, device=device
    )
    entry_mib = torch.cuda.memory_allocated(device) / 1024**2
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        projected = _project_msa_features_chunked(
            projection,
            *features,
            use_paired_feature=True,
            num_tokens=const.num_tokens,
            sequence_chunk=args.sequence_chunk,
        )
    torch.cuda.synchronize(device)
    expected_shape = (1, args.num_sequences, args.num_tokens, args.channels)
    if projected.shape != expected_shape:
        raise AssertionError(f"unexpected output shape: {projected.shape}")
    if projected.dtype != torch.bfloat16:
        raise AssertionError(f"unexpected output dtype: {projected.dtype}")
    projection_allocated_mib = torch.cuda.memory_allocated(device) / 1024**2
    projection_peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    for sequence_start in range(0, projected.shape[1], args.sequence_chunk):
        sequence_end = min(
            sequence_start + args.sequence_chunk, projected.shape[1]
        )
        if not torch.isfinite(projected[:, sequence_start:sequence_end]).all():
            raise AssertionError("projected MSA contains non-finite values")

    print(
        "MSA_PROJECTION_FULL_SHAPE_PASSED "
        f"shape={tuple(projected.shape)} dtype={projected.dtype} "
        f"entry_mib={entry_mib:.1f} allocated_mib={projection_allocated_mib:.1f} "
        f"peak_mib={projection_peak_mib:.1f}"
    )
    del floor_reservation


if __name__ == "__main__":
    main()