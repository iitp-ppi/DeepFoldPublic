# Copyright 2024 DeepFold Team


from typing import Any, Tuple

import torch
import torch.distributed

import deepfold.core.parallel_state as ps
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
    tensor = tensor.contiguous()

    if dim == 1 and tensor.size(0) == 1:
        # Tensors in the list are contiguous partst of the output
        output_shape = list(tensor.shape)
        output_shape[1] *= world_size
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = list(output.chunk(world_size, dim=1))
        torch.distributed.all_gather(
            tensor_list=tensor_list,
            tensor=tensor,
            group=get_model_parallel_group(),
            async_op=False,
        )
    else:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(
            tensor_list=tensor_list,
            tensor=tensor,
            group=get_model_parallel_group(),
            async_op=False,
        )
        output = torch.cat(tensor_list, dim=dim)

    return output


def _all_reduce_sum_split(tensor: torch.Tensor, dim: int) -> torch.Tensor:

    world_szie = get_model_parallel_world_size()
    rank = get_model_parallel_rank()
    tensor = tensor.contiguous()

    torch.distributed.all_reduce(
        tensor=tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=get_model_parallel_group(),
    )

    assert tensor.size(dim) % world_szie == 0
    chunks = tensor.chunk(world_szie, dim=dim)
    output = chunks[rank]

    return output


def _split(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split the tensor along the dimension and keep the corresponding slice."""

    world_size = get_model_parallel_world_size()
    assert tensor.size(dim) % world_size == 0

    # chunk = split_tensor(tensor, world_size, dim=dim)
    chunk = tensor.chunk(world_size, dim=dim)

    rank = get_model_parallel_rank()
    output = chunk[rank].contiguous()

    return output


def _all_to_all(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """
    Scatter and gather the tensor.

    Note:
        `dim0` should be sharded and `dim1` should be whole.
        The result tensor is sharded in `dim1`.
    """

    world_size = get_model_parallel_world_size()
    assert tensor.size(dim0) % world_size == 0

    input_tensor_list = [input_tensor.contiguous() for input_tensor in tensor.chunk(world_size, dim=dim0)]

    if dim1 == 1 and tensor.size(0) == 1:
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= world_size
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = list(output.chunk(world_size, dim=1))
        torch.distributed.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=get_model_parallel_group(),
            async_op=False,
        )
    else:
        output_tensor_list = [torch.empty_like(input_tensor) for input_tensor in input_tensor_list]
        torch.distributed.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=get_model_parallel_group(),
            async_op=False,
        )
        output = torch.cat(output_tensor_list, dim=dim1)

    return output


def _broadcast(tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    """
    Broadcast a tensor to whole group ranks.
    """

    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return tensor

    torch.distributed.broadcast(tensor, src_rank, group=get_model_parallel_group())

    return tensor


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


class _GatherAllReduceSumFromModelParallelRegion(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: "_GatherAllReduceSumFromModelParallelRegion",
        input: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor([dim]))
        return _gather(input, dim=dim)

    @staticmethod
    def backward(
        ctx: "_GatherAllReduceSumFromModelParallelRegion",
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        return _all_reduce_sum_split(grad_output, dim=int(ctx.saved_tensors[0][0])), None


class _TransposeOnModelParallelRegion(torch.autograd.Function):
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
        return _all_to_all(grad_output, dim1, dim0), None, None


class _BroadcastOnModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, src_rank: int):
        ctx._src_rank = src_rank
        return _broadcast(tensor, src_rank)

    def backward(ctx, grad_output):
        grad_reduced = _reduce(grad_output)  # TODO: Isn't it 'mean'?
        if get_model_parallel_rank() != ctx._src_rank:
            grad_reduced *= 0.0
        return grad_reduced, None


#
# Helpers
#


def copy_to_model_parallel_reigon(tensor: torch.Tensor) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(tensor)


def reduce_from_model_parallel_region(tensor: torch.Tensor) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(tensor)


reduce = reduce_from_model_parallel_region


def scatter_to_model_parallel_region(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    if not ps.is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = _ScatterToModelParallelRegion.apply(tensor, dim)
    else:
        tensor = _split(tensor, dim=dim)

    return tensor


scatter = scatter_to_model_parallel_region


def gather_from_model_parallel_region(tensor: torch.Tensor, dim: int, bwd: str = "split") -> torch.Tensor:
    if not ps.is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        if bwd == "split":
            tensor = _GatherFromModelParallelRegion.apply(tensor, dim)
        elif bwd == "all_reduce_sum_split":
            tensor = _GatherAllReduceSumFromModelParallelRegion.apply(tensor, dim)
        else:
            raise ValueError(f"Unknown bwd={repr(bwd)}")
    else:
        tensor = _gather(tensor, dim=dim)

    return tensor


gather = gather_from_model_parallel_region


def transpose_on_model_parallel_region(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    return _TransposeOnModelParallelRegion.apply(tensor, dim0, dim1)


all_to_all = transpose_on_model_parallel_region


def broadcast_on_model_parallel_region(tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    return _BroadcastOnModelParallelRegion.apply(tensor, src_rank)


broadcast = broadcast_on_model_parallel_region


def col_to_row(tensor: torch.Tensor) -> torch.Tensor:
    if not ps.is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = _TransposeOnModelParallelRegion.apply(tensor, -3, -2)
    else:
        tensor = _all_to_all(tensor, -3, -2)

    return tensor


def row_to_col(tensor: torch.Tensor) -> torch.Tensor:
    if not ps.is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = _TransposeOnModelParallelRegion.apply(tensor, -2, -3)
    else:
        tensor = _all_to_all(tensor, -2, -3)

    return tensor
