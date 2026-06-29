"""Runtime dtype guards for DAP communication hot paths."""

from __future__ import annotations

import torch
from torch import Tensor

_LOGGED_BF16_GUARDS: set[str] = set()
PAIR_STORAGE_DTYPE = torch.bfloat16


def get_pair_storage_dtype() -> torch.dtype:
    """Return the canonical persistent dtype for DAP pair activations."""
    return PAIR_STORAGE_DTYPE


def restore_pair_storage_dtype(
    tensor: Tensor,
    *,
    dtype: torch.dtype = PAIR_STORAGE_DTYPE,
) -> Tensor:
    """Keep persistent pair activations in the DAP communication dtype.

    DAP pair sub-ops may compute in float32 internally, but residual streams
    should be stored as bf16 before any all-to-all boundary to avoid carrying
    double-sized buffers into ``row_to_col`` / ``col_to_row``.
    """
    if tensor.dtype != dtype:
        return tensor.to(dtype=dtype)
    return tensor


def ensure_bf16_row_to_col_input(
    tensor: Tensor,
    *,
    tag: str,
    rank: int,
) -> None:
    """Fail fast unless a pair activation is bf16 before ``row_to_col``.

    We keep some numerically sensitive ops in float32 internally, but the
    persistent pair activations that enter DAP all-to-all should be stored as
    bf16 to keep communication buffers within VRAM limits.
    """
    if tensor.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{tag}: row_to_col expected torch.bfloat16 input, got {tensor.dtype}"
        )
    if rank == 0 and tag not in _LOGGED_BF16_GUARDS:
        print(
            f"[DTYPE-GUARD] {tag}: verified row_to_col input dtype {tensor.dtype}",
            flush=True,
        )
        _LOGGED_BF16_GUARDS.add(tag)
