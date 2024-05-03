# Copyright 2024 DeepFold Team


from __future__ import annotations

from typing import Tuple

import torch

import deepfold.distributed.main as dist

# whether DAP has been initialized or not
_DAP_INITIALIZED = False

# whether DAP is enabled or not
_DAP_ENABLED = None

# DAP size, one of: 0, 1, 2, 4 or 8
_DAP_SIZE = 0

# DAP process group
_DAP_GROUP = None

# DAP group rank: from 0 to num_dap_groups-1
_DAP_GROUP_RANK = None

# process rank inside DAP group: from 0 to dap_size-1
_DAP_RANK = None


def initialize(dap_size: int) -> None:
    """
    Initialize Dynamic Axial Parallelism (DAP).

    Args:
        dap_size: number of GPUs used in DAP group.

    """
    global _DAP_INITIALIZED
    global _DAP_ENABLED
    global _DAP_SIZE
    global _DAP_GROUP
    global _DAP_GROUP_RANK
    global _DAP_RANK

    assert not _DAP_INITIALIZED
    assert _DAP_ENABLED is None
    assert _DAP_SIZE == 0
    assert _DAP_GROUP is None
    assert _DAP_GROUP_RANK is None
    assert _DAP_RANK is None
    # assert dap_size in {1, 2, 4, 8}
    assert dist.is_initialized()

    num_train_ranks = len(dist.train_ranks())
    if num_train_ranks % dap_size != 0:
        raise RuntimeError(f"num_train_ranks={num_train_ranks} is not divisible by dap_size={dap_size}")
    num_dap_groups = num_train_ranks // dap_size

    for dap_group_rank in range(num_dap_groups):
        ranks_forming_dap_group = list(
            range(
                dap_group_rank * dap_size,
                (dap_group_rank + 1) * dap_size,
            ),
        )
        group = torch.distributed.new_group(ranks_forming_dap_group)
        if dist.rank() in ranks_forming_dap_group:
            _DAP_GROUP = group
            assert dap_group_rank == dist.rank() // dap_size
            _DAP_GROUP_RANK = dap_group_rank
            _DAP_RANK = dist.rank() % dap_size

    _DAP_SIZE = dap_size
    _DAP_ENABLED = True
    _DAP_INITIALIZED = True


def is_initialized() -> bool:
    return _DAP_INITIALIZED


def is_enabled() -> bool:
    return bool(_DAP_ENABLED)


def size() -> int:
    return _DAP_SIZE


def group() -> torch.distributed.ProcessGroup:
    assert _DAP_INITIALIZED
    return _DAP_GROUP


def group_rank() -> int:
    assert _DAP_INITIALIZED
    return _DAP_GROUP_RANK


def rank() -> int:
    assert _DAP_INITIALIZED
    return _DAP_RANK


def _enable() -> None:
    global _DAP_ENABLED
    _DAP_ENABLED = True


def _disable() -> None:
    global _DAP_ENABLED
    _DAP_ENABLED = False


class Enable(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Enable) -> None:
        return _enable()

    @staticmethod
    def backward(ctx: Enable) -> None:
        return _disable()


class Disable(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Disable) -> None:
        return _disable()

    @staticmethod
    def backward(ctx: Disable) -> None:
        return _enable()


def enable() -> None:
    if is_initialized():
        if torch.is_grad_enabled():
            Enable.apply()
        else:
            _enable()


def disable() -> None:
    if is_initialized():
        if torch.is_grad_enabled():
            Disable.apply()
        else:
            _disable()


def _reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    # All-reduce
    torch.distributed.all_reduce(tensor, group=_DAP_GROUP)

    return tensor


def _gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""

    tensor = tensor.contiguous()

    if dim == 1 and tensor.size(0) == 1:
        # Tensors in the list are contiguous partst of the output
        output_shape = list(tensor.shape)
        output_shape[1] *= _DAP_SIZE
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        tensor_list = list(output.chunk(_DAP_SIZE, dim=1))
        torch.distributed.all_gather(
            tensor_list=tensor_list,
            tensor=tensor,
            group=_DAP_GROUP,
            async_op=False,
        )
    else:
        tensor_list = [torch.empty_like(tensor) for _ in range(_DAP_SIZE)]
        torch.distributed.all_gather(
            tensor_list=tensor_list,
            tensor=tensor,
            group=_DAP_GROUP,
            async_op=False,
        )
        output = torch.cat(tensor_list, dim=dim)

    return output


def _all_reduce_sum_split(tensor: torch.Tensor, dim: int) -> torch.Tensor:

    tensor = tensor.contiguous()

    torch.distributed.all_reduce(
        tensor=tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=_DAP_GROUP,
    )

    assert tensor.size(dim) % _DAP_SIZE == 0
    chunks = tensor.chunk(_DAP_SIZE, dim=dim)
    output = chunks[()]

    return output


def _split(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Split the tensor along the dimension and keep the corresponding slice."""

    assert tensor.size(dim) % _DAP_SIZE == 0
    chunk = tensor.chunk(_DAP_SIZE, dim=dim)
    output = chunk[_DAP_RANK].contiguous()

    return output


def _all_to_all(tensor: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """
    Scatter and gather the tensor.

    Note:
        `dim0` should be sharded and `dim1` should be whole.
        The result tensor is sharded in `dim1`.
    """

    assert tensor.size(dim0) % _DAP_SIZE == 0

    input_tensor_list = [input_tensor.contiguous() for input_tensor in tensor.chunk(_DAP_SIZE, dim=dim0)]

    if dim1 == 1 and tensor.size(0) == 1:
        output_shape = list(input_tensor_list[0].shape)
        output_shape[1] *= _DAP_SIZE
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        output_tensor_list = list(output.chunk(_DAP_SIZE, dim=1))
        torch.distributed.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=_DAP_GROUP,
            async_op=False,
        )
    else:
        output_tensor_list = [torch.empty_like(input_tensor) for input_tensor in input_tensor_list]
        torch.distributed.all_to_all(
            output_tensor_list=output_tensor_list,
            input_tensor_list=input_tensor_list,
            group=_DAP_GROUP,
            async_op=False,
        )
        output = torch.cat(output_tensor_list, dim=dim1)

    return output


def _broadcast(tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    """
    Broadcast a tensor to whole group ranks.
    """

    if _DAP_SIZE == 1:
        return tensor

    torch.distributed.broadcast(tensor, src_rank, group=_DAP_GROUP)

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
    def backward(ctx, grad_output):
        dim0, dim1 = ctx._dim0, ctx._dim1
        return _all_to_all(grad_output, dim1, dim0), None, None


class _BroadcastOnModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, src_rank: int):
        ctx._src_rank = src_rank
        return _broadcast(tensor, src_rank)

    def backward(ctx, grad_output):
        grad_reduced = _reduce(grad_output)
        if _DAP_RANK != ctx._src_rank:
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
    if not is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = _ScatterToModelParallelRegion.apply(tensor, dim)
    else:
        tensor = _split(tensor, dim=dim)

    return tensor


scatter = scatter_to_model_parallel_region


def gather_from_model_parallel_region(tensor: torch.Tensor, dim: int, bwd: str = "split") -> torch.Tensor:
    if not is_enabled():
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
    if not is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = _TransposeOnModelParallelRegion.apply(tensor, -3, -2)
    else:
        tensor = _all_to_all(tensor, -3, -2)

    return tensor


def row_to_col(tensor: torch.Tensor) -> torch.Tensor:
    if not is_enabled():
        return tensor

    if torch.is_grad_enabled() and tensor.requires_grad:
        tensor = _TransposeOnModelParallelRegion.apply(tensor, -2, -3)
    else:
        tensor = _all_to_all(tensor, -2, -3)

    return tensor
