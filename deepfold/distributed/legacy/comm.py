# Copyright 2023 Deepfold Team

from typing import Any, List, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as dist
from torch import Tensor

from deepfold.distributed.legacy.core import (
    TENSOR_MODEL_PARALLEL_GROUP,
    _ensure_divisibility,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from deepfold.utils.debug import dump_args

_AsyncHandle = Optional[dist.Work]


def _alone() -> bool:
    return get_tensor_model_parallel_world_size() == 1


def _divide(n: int, d: int) -> int:
    _ensure_divisibility(n, d)
    return n // d


# Primitive synchronous communication routines


# All-redcue element-wise
def _reduce(tensor: Tensor, op: str = "sum") -> Tensor:
    if _alone():
        return tensor

    op = op.lower()

    if op == "sum":
        op = dist.ReduceOp.SUM
    elif op == "prod":
        op = dist.ReduceOp.PRODUCT
    elif op == "min":
        op = dist.ReduceOp.MIN
    elif op == "max":
        op = dist.ReduceOp.MAX
    else:
        ValueError(f"{op} is not supported reduce operation")

    # All-reduce
    dist.all_reduce(
        tensor,
        op=op,
        group=TENSOR_MODEL_PARALLEL_GROUP,
        async_op=False,
    )

    return tensor


# Split a tensor
def _split(tensor: Tensor, dim: int = -1) -> Tensor:
    if _alone():
        return tensor

    split_size = _divide(tensor.shape[dim], get_tensor_model_parallel_world_size())
    tensor_list = torch.split(tensor, split_size, dim=dim)

    out = tensor_list[get_tensor_model_parallel_rank()].contiguous()

    return out


# All-gather
def _gather(tensor: Tensor, dim: int = -1) -> Tensor:
    if _alone():
        return tensor

    # Allocate a buffer.
    tensor_list = [torch.empty_like(tensor) for _ in range(get_tensor_model_parallel_world_size())]
    tensor = tensor.contiguous()
    # All-gather.
    dist.all_gather(tensor_list, tensor, group=TENSOR_MODEL_PARALLEL_GROUP, async_op=False)
    # Target dimension will be expanded.
    out = torch.cat(tensor_list, dim=dim)

    return out


# Identity


class Identity(torch.autograd.Function):
    """
    Class for backward reduce operation.
    It does not communicate forwardly.
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        return _reduce(grad_output)


@dump_args
def identity(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Identity.apply(input)
    return input


# Scatter


class Scatter(torch.autograd.Function):
    """
    Class for backward scatter operation.
    It does not communicate forwardly.
    """

    @staticmethod
    def forward(ctx: Any, input: Tensor, dim: int = -1) -> Tensor:
        ctx._dim = dim
        return _split(input, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        dim = ctx._dim
        return _gather(grad_output, dim=dim), None


@dump_args
def scatter(input: Tensor, dim: int = -1) -> Tensor:
    """
    Scatter a tensor.
    """
    if torch.is_grad_enabled() and input.requires_grad:
        input = Scatter.apply(input, dim)
    else:
        input = _split(input, dim=dim)
    return input


# All-Reduce


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        return _reduce(input, op="sum")

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tensor:
        return grad_output


@dump_args
def reduce(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Reduce.apply(input)
    else:
        input = _reduce(input, op="sum")
    return input


# All-Gather


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, dim: int = -1) -> Tensor:
        ctx._dim = dim
        return _gather(input, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        dim = ctx._dim
        return _split(grad_output, dim=dim), None


@dump_args
def gather(input: Tensor, dim: int = -1) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = Gather.apply(input, dim)
    else:
        input = _gather(input, dim=dim)
    return input


# Synchronized transpose routines (All to All)


def _all_to_all_sync(tensor: Tensor, in_dim: int, out_dim: int) -> Tensor:
    if _alone():
        return tensor

    tensor = tensor.transpose(in_dim, 0).contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=TENSOR_MODEL_PARALLEL_GROUP)
    output = output.transpose(in_dim, 0).contiguous()
    tensor_list = output.chunk(get_tensor_model_parallel_world_size(), dim=in_dim)
    return torch.cat(tensor_list, dim=out_dim)


@dump_args
def col_to_row(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = All_to_All.apply(input, -3, -2)
    else:
        input = _all_to_all_sync(input, in_dim=-3, out_dim=-2)
    return input


@dump_args
def row_to_col(input: Tensor) -> Tensor:
    if torch.is_grad_enabled() and input.requires_grad:
        input = All_to_All.apply(input, -2, -3)
    else:
        input = _all_to_all_sync(input, in_dim=-2, out_dim=-3)
    return input


class All_to_All(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, in_dim: int, out_dim: int) -> Tensor:
        ctx._in_dim = in_dim
        ctx._out_dim = out_dim
        return _all_to_all_sync(input, in_dim, out_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        in_dim = ctx._in_dim
        out_dim = ctx._out_dim
        return _all_to_all_sync(grad_output, in_dim=out_dim, out_dim=in_dim), None, None


# Synchronized Broadcast


@dump_args
def broadcast_sync(src_rank: int, tensor: Tensor, host: bool) -> Tensor:
    """
    Broadcasts the tensor to the whole group.

    Args:
        src_rank (int):
            Local source rank.
        tensor (Tensor):
            Data to be sent.
    Returns:
        None if src.
        Tensor if not src.
    """
    if _alone():
        return None

    if host:
        dist.broadcast(tensor, src_rank, group=TENSOR_MODEL_PARALLEL_GROUP, async_op=False)
        return None
    else:
        out = torch.empty_like(tensor)
        dist.broadcast(out, src=src_rank, group=TENSOR_MODEL_PARALLEL_GROUP, async_op=False)
        return out


# Asynchronous Broadcast


def broadcast_async_begin(src_rank: int, tensor: Tensor, host: bool) -> _AsyncHandle:
    if _alone():
        return None

    if host:
        work = dist.broadcast(tensor, src=src_rank, group=TENSOR_MODEL_PARALLEL_GROUP, async_op=True)
    else:
        work = dist.broadcast(tensor, src=src_rank, group=TENSOR_MODEL_PARALLEL_GROUP, async_op=True)
    return work


def broadcast_async_end(work: _AsyncHandle) -> None:
    if work:
        work.wait()


# Asynchronous All-Gather


def _gather_async(tensor: Tensor, dim: int = -1) -> Tuple[Tensor, _AsyncHandle]:
    if _alone():
        return tensor, None

    world_size = get_tensor_model_parallel_world_size()
    output_shape = list(tensor.shape)
    dim = dim % tensor.ndim
    output_shape[dim] *= world_size
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    tensor_list = output.chunk(world_size, dim=dim)
    tensor_list = [tensor.contiguous() for tensor in tensor_list]

    # Async all-gather
    work = dist.all_gather(list(tensor_list), tensor, group=TENSOR_MODEL_PARALLEL_GROUP, async_op=True)

    return output, work


class GatherAsyncBegin(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor, dim: int = -1) -> Tuple[Tensor, _AsyncHandle]:
        ctx._dim = dim
        return _gather_async(input, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor, grad_work=None) -> Tuple[Tensor]:
        dim = ctx._dim
        return _split(grad_output, dim=dim).squeeze(dim=dim), None


class GatherAsyncEnd(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input: Tensor) -> Tensor:
        # ctx._dim = dim
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor]:
        return grad_output, None


def gather_async_begin(input: Tensor, dim: int = -1) -> Tuple[Tensor, _AsyncHandle]:
    if torch.is_grad_enabled() and input.requires_grad:
        input, work = GatherAsyncBegin.apply(input, dim)
    else:
        input, work = _gather_async(input, dim=dim)
    return input, work


def gather_async_end(input: List[Tensor], work: _AsyncHandle) -> Tensor:
    if work:
        work.wait()
    if torch.is_grad_enabled() and input[0].requires_grad:
        output = GatherAsyncEnd.apply(input)
    else:
        output = input
    return output
