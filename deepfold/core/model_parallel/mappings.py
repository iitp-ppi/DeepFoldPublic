# Copyright 2024 DeepFold Team


from typing import Any

import torch
import torch.distributed

from deepfold.core.model_parallel.utils import split_tensor
from deepfold.core.parallel_state import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
)


def _reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # Bypass
    if get_model_parallel_world_size() == 1:
        return tensor

    # All-reduce
    torch.distributed.all_reduce(tensor, group=get_model_parallel_group())

    return tensor


def _gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""

    world_size = get_model_parallel_world_size()
    # Bypass
    if world_size == 1:
        return tensor

    ndim = tensor.dim()
    dim = dim + ndim if dim < 0 else dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"Dim {dim} is out of bound")

    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]

    # All-gather
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), group=get_model_parallel_group())

    # Concatenate
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _split(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split the tensor along the dimension and keep the corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass
    if world_size == 1:
        return tensor

    # Split
    tensor_list = split_tensor(tensor, world_size, dim=dim)

    rank = get_model_parallel_rank()
    output = tensor_list[rank].contiguous()

    return output


def _all_to_all(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """
    Scatter and gather the tensor.

    Note:
        `dim0` should be sharded and `dim1` should be whole.
        The result tensor is sharded in `dim1`.
    """

    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return tensor

    send_buffer = tensor.contiguous()
    recv_buffer = torch.empty_like(send_buffer)

    intput_tensors = send_buffer.chunk(world_size, dim=dim1)
    output_tensors = recv_buffer.chunk(world_size, dim=dim1)

    torch.distributed.all_to_all(output_tensors, intput_tensors, group=get_model_parallel_group())

    return torch.cat(output_tensors, dim=dim0)


#
# Functions
#


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return _reduce(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chunk to the rank."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx._dim = dim
        return _split(tensor, dim=dim)

    def backward(ctx, grad_output):
        dim = ctx._dim
        return _gather(grad_output, dim=dim), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """All-gather the input from model parallel region and concatenate."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        ctx._dim = dim
        return _gather(tensor, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim = ctx._dim
        return _split(grad_output, dim=dim), None


class _TransposeToModelParallelRegion(torch.autograd.Function):
    """
    Transpose the input with the given dimensions.

    Note:
        Assume that the input is properly distributed (in the first dimension).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
        ctx._dim0, ctx._dim1 = dim0, dim1
        return _all_to_all(tensor, dim0, dim1)

    @staticmethod
    def backward(ctx: Any, grad_output):
        dim0, dim1 = ctx._dim0, ctx._dim1
        return _all_to_all(grad_output, dim0, dim1), None, None


#
# Helpers
#


def copy_to_model_parallel_reigon(tensor: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(tensor)


def reduce_from_model_parallel_region(tensor: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(tensor)


def scatter_to_model_parallel_region(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _ScatterToModelParallelRegion.apply(tensor, dim)


def gather_from_model_parallel_region(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _GatherFromModelParallelRegion.apply(tensor, dim)


def transpose_to_model_parallel_region(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return _TransposeToModelParallelRegion(tensor, dim0, dim1)
