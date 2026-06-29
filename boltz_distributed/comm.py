"""
DAP Communication Primitives for Boltz

Adapted from FastFold's fastfold/distributed/comm.py

These primitives handle tensor distribution across GPUs:
- scatter: Split tensor across GPUs
- gather: Collect tensor shards from all GPUs
- row_to_col / col_to_row: All-to-all for attention axis swaps
"""

from typing import Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .core import get_dap_size, get_dap_rank, get_dap_group, ensure_divisibility


def divide(numerator, denominator):
    """Divide, rounding up to handle non-divisible sizes."""
    return (numerator + denominator - 1) // denominator


def _reduce(tensor: Tensor) -> Tensor:
    """All-reduce operation across DAP group."""
    dap_size = get_dap_size()
    if dap_size == 1:
        return tensor

    group = get_dap_group()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=False)
    return tensor


def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    """Split tensor along dimension and keep only local shard.
    
    Handles non-divisible sizes by padding, then removing padding on the last rank.
    """
    dap_size = get_dap_size()
    if dap_size == 1:
        return tensor

    dap_rank = get_dap_rank()
    size = tensor.shape[dim]
    
    # Calculate split size (round up)
    split_size = divide(size, dap_size)
    
    # Pad if necessary
    padded_size = split_size * dap_size
    if padded_size > size:
        pad_amount = padded_size - size
        # Create padding shape
        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_amount
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=dim)
    
    # Split and take local shard
    tensor_list = torch.split(tensor, split_size, dim=dim)
    output = tensor_list[dap_rank].contiguous()
    
    return output


def _gather(tensor: Tensor, dim: int = -1, original_size: int = None) -> Tensor:
    """Gather tensor shards from all GPUs.
    
    If original_size is provided, trims the result to that size (removes padding).
    """
    dap_size = get_dap_size()
    if dap_size == 1:
        return tensor

    group = get_dap_group()
    
    if dim == 1 and list(tensor.shape)[0] == 1:
        output_shape = list(tensor.shape)
        output_shape[1] *= dap_size
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = output.chunk(dap_size, dim=1)
        dist.all_gather(list(tensor_list), tensor, group=group, async_op=False)
    else:
        tensor_list = [torch.empty_like(tensor) for _ in range(dap_size)]
        dist.all_gather(tensor_list, tensor, group=group, async_op=False)
        output = torch.cat(tensor_list, dim=dim)
    
    # Trim to original size if provided (remove padding)
    if original_size is not None and output.shape[dim] > original_size:
        indices = [slice(None)] * len(output.shape)
        indices[dim] = slice(0, original_size)
        output = output[tuple(indices)]

    return output


def _trim_dim(tensor: Tensor, dim: int, original_size: int | None) -> Tensor:
    if original_size is None or tensor.shape[dim] <= original_size:
        return tensor
    indices = [slice(None)] * len(tensor.shape)
    indices[dim] = slice(0, original_size)
    return tensor[tuple(indices)]


def _gather_to_rank0_cpu(
    tensor: Tensor,
    dim: int = -1,
    original_size: int | None = None,
    root: int = 0,
    pin_memory: bool = False,
) -> Tensor | None:
    """Gather shards to ``root`` CPU memory without full-rank output tensors."""
    dap_size = get_dap_size()
    dap_rank = get_dap_rank()
    if dap_size == 1:
        output = _trim_dim(tensor.detach().cpu(), dim, original_size)
        return output.pin_memory() if pin_memory else output

    group = get_dap_group()
    send_tensor = tensor.detach().contiguous()
    output = None
    if dap_rank == root:
        shards = [None] * dap_size
        shards[root] = send_tensor.cpu()
        for src in range(dap_size):
            if src == root:
                continue
            meta = torch.empty(1, dtype=torch.int64, device=send_tensor.device)
            dist.recv(meta, src=src, group=group)
            payload = torch.empty(
                int(meta.item()), dtype=send_tensor.dtype, device=send_tensor.device
            )
            dist.recv(payload, src=src, group=group)
            shards[src] = payload.reshape_as(send_tensor).cpu()
        output = torch.cat(shards, dim=dim)
        output = _trim_dim(output, dim, original_size)
        if pin_memory:
            output = output.pin_memory()
    else:
        payload = send_tensor.reshape(-1)
        meta = torch.tensor([payload.numel()], dtype=torch.int64, device=send_tensor.device)
        dist.send(meta, dst=root, group=group)
        dist.send(payload, dst=root, group=group)
    return output


# Autograd-compatible wrappers with proper gradient handling

class Copy(torch.autograd.Function):
    """Copy operation - identity forward, reduce backward."""
    
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return _reduce(grad_output)


class Scatter(torch.autograd.Function):
    """Scatter operation - split forward, gather backward."""
    
    @staticmethod
    def forward(ctx, input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        dim, = ctx.saved_tensors
        return _gather(grad_output, dim=int(dim)), None


class Reduce(torch.autograd.Function):
    """Reduce operation - all-reduce forward, identity backward."""
    
    @staticmethod
    def forward(ctx, input: Tensor) -> Tensor:
        return _reduce(input)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


class Gather(torch.autograd.Function):
    """Gather operation - gather forward, split backward."""
    
    @staticmethod
    def forward(ctx, input: Tensor, dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        dim, = ctx.saved_tensors
        return _split(grad_output, dim=int(dim)), None


def _all_to_all(tensor: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
    """All-to-all operation for dimension swapping.
    
    Handles non-divisible sizes by using split_size that works with the padded tensor.
    """
    dap_size = get_dap_size()
    if dap_size == 1:
        return tensor

    group = get_dap_group()
    
    # The tensor should already be padded from scatter, so it should be divisible
    size = tensor.shape[in_dim]
    split_size = divide(size, dap_size)
    
    # Handle case where size is already padded and divisible
    if split_size * dap_size != size:
        # Need to pad
        pad_amount = split_size * dap_size - size
        pad_shape = list(tensor.shape)
        pad_shape[in_dim] = pad_amount
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.cat([tensor, padding], dim=in_dim)
    
    input_tensor_list = torch.split(tensor, split_size, dim=in_dim)
    input_tensor_list = [t.contiguous() for t in input_tensor_list]

    if out_dim == 1:
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= dap_size
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = output.chunk(dap_size, dim=1)
        dist.all_to_all(list(output_tensor_list), input_tensor_list, group=group, async_op=False)
    else:
        output_tensor_list = [torch.ones_like(t) for t in input_tensor_list]
        dist.all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=False)
        output = torch.cat(output_tensor_list, dim=out_dim)

    return output


class All_to_All(torch.autograd.Function):
    """All-to-all operation with autograd support."""
    
    @staticmethod
    def forward(ctx, input: Tensor, in_dim: int = -1, out_dim: int = -1) -> Tensor:
        ctx.save_for_backward(torch.tensor([in_dim, out_dim]))
        return _all_to_all(input, in_dim=in_dim, out_dim=out_dim)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None]:
        saved_tensors = ctx.saved_tensors[0]
        return _all_to_all(
            grad_output, 
            in_dim=int(saved_tensors[1]), 
            out_dim=int(saved_tensors[0])
        ), None, None


# Public API with gradient support

def copy(input: Tensor) -> Tensor:
    """Copy tensor (identity forward, reduce backward for gradients)."""
    if torch.is_grad_enabled() and input.requires_grad:
        return Copy.apply(input)
    return input


def scatter(input: Tensor, dim: int = -1) -> Tensor:
    """Scatter tensor across GPUs along specified dimension."""
    if torch.is_grad_enabled() and input.requires_grad:
        return Scatter.apply(input, dim)
    return _split(input, dim=dim)


def reduce(input: Tensor) -> Tensor:
    """All-reduce tensor across GPUs."""
    if torch.is_grad_enabled() and input.requires_grad:
        return Reduce.apply(input)
    return _reduce(input)


def gather(input: Tensor, dim: int = -1, original_size: int = None) -> Tensor:
    """Gather tensor from all GPUs along specified dimension.
    
    If original_size is provided, trims the result to remove padding.
    """
    if torch.is_grad_enabled() and input.requires_grad:
        output = Gather.apply(input, dim)
        # Trim padding if needed
        if original_size is not None and output.shape[dim] > original_size:
            indices = [slice(None)] * len(output.shape)
            indices[dim] = slice(0, original_size)
            output = output[tuple(indices)]
        return output
    return _gather(input, dim=dim, original_size=original_size)


def gather_to_rank0_cpu(
    input: Tensor,
    dim: int = -1,
    original_size: int = None,
    root: int = 0,
    pin_memory: bool = False,
) -> Tensor | None:
    """Gather tensor shards to root CPU memory only.

    This is for inference-only memory exits where non-root ranks must
    participate in communication but must not allocate the reconstructed tensor.
    """
    if torch.is_grad_enabled() and input.requires_grad:
        raise RuntimeError("gather_to_rank0_cpu is inference-only")
    return _gather_to_rank0_cpu(
        input,
        dim=dim,
        original_size=original_size,
        root=root,
        pin_memory=pin_memory,
    )


def col_to_row(input: Tensor) -> Tensor:
    """Convert column-distributed to row-distributed tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        return All_to_All.apply(input, 1, 2)
    return _all_to_all(input, in_dim=1, out_dim=2)


def row_to_col(input: Tensor) -> Tensor:
    """Convert row-distributed to column-distributed tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        return All_to_All.apply(input, 2, 1)
    return _all_to_all(input, in_dim=2, out_dim=1)


# ── Async gather primitives (FastFold-style) ────────────────────────────

def gather_async(tensor: Tensor, dim: int = -1):
    """Start a non-blocking all_gather and return (output_list, work_handle).
    
    Allows overlapping communication with computation:
        out_list, handle = gather_async(x, dim=1)
        # ... do other compute here ...
        result = gather_async_opp(out_list, handle, dim=1)
    """
    dap_size = get_dap_size()
    if dap_size == 1:
        return [tensor], None

    group = get_dap_group()
    tensor_list = [torch.empty_like(tensor) for _ in range(dap_size)]
    work = dist.all_gather(tensor_list, tensor.contiguous(), group=group, async_op=True)
    return tensor_list, work


def gather_async_opp(tensor_list, work, dim: int = -1, original_size: int = None) -> Tensor:
    """Wait for async gather to complete and concatenate results.
    
    Args:
        tensor_list: List of tensors from gather_async
        work: Work handle from gather_async (None if dap_size==1)
        dim: Dimension to concatenate along
        original_size: If provided, trim to this size (remove padding)
    """
    if work is not None:
        work.wait()

    output = torch.cat(tensor_list, dim=dim)

    if original_size is not None and output.shape[dim] > original_size:
        indices = [slice(None)] * len(output.shape)
        indices[dim] = slice(0, original_size)
        output = output[tuple(indices)]

    return output
