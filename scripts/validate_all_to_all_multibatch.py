"""Validate DAP row/column all-to-all for single- and multi-batch tensors."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from boltz_distributed.comm import (
    col_to_row,
    col_to_row_inplace,
    gather,
    row_to_col,
    scatter,
)
from boltz_distributed.core import get_dap_rank, get_dap_size, init_dap


def validate_round_trip(batch_size: int, n_tokens: int = 8, channels: int = 4) -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    numel = batch_size * n_tokens * n_tokens * channels
    full = torch.arange(numel, dtype=torch.float32, device=device).reshape(
        batch_size, n_tokens, n_tokens, channels
    )

    with torch.no_grad():
        row_shard = scatter(full, dim=1)
        col_shard = row_to_col(row_shard)
        gathered_col = gather(col_shard, dim=2, original_size=n_tokens)
        torch.testing.assert_close(
            gathered_col[:, :n_tokens], full, rtol=0, atol=0
        )

        row_round_trip = col_to_row(col_shard)
        row_inplace = row_shard.clone()
        output_ptr = row_inplace.data_ptr()
        returned = col_to_row_inplace(col_shard, row_inplace)
        if returned.data_ptr() != output_ptr:
            raise AssertionError("col_to_row_inplace replaced its output storage")
        torch.testing.assert_close(
            row_inplace,
            row_round_trip[:, :, :n_tokens],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            gather(row_round_trip, dim=1, original_size=n_tokens)[:, :, :n_tokens],
            full,
            rtol=0,
            atol=0,
        )

    if n_tokens % get_dap_size() == 0:
        full_grad = full.clone().requires_grad_(True)
        row_round_trip = col_to_row(row_to_col(scatter(full_grad, dim=1)))
        row_round_trip.square().sum().backward()
        torch.testing.assert_close(full_grad.grad, 2 * full_grad, rtol=0, atol=0)

    if get_dap_rank() == 0:
        print(
            f"batch_size={batch_size}, n_tokens={n_tokens}: "
            "allocating/in-place round-trip passed",
            flush=True,
        )


def main() -> None:
    init_dap()
    if get_dap_size() != 2:
        raise RuntimeError(f"Expected 2 DAP ranks, got {get_dap_size()}")

    validate_round_trip(batch_size=1)
    validate_round_trip(batch_size=2)
    validate_round_trip(batch_size=1, n_tokens=9)
    validate_round_trip(batch_size=2, n_tokens=9)
    dist.barrier()
    if get_dap_rank() == 0:
        print("Multi-batch all-to-all validation passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()