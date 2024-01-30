# Copyright 2024 DeepFold Team


from typing import List

import torch
import torch.distributed

from deepfold.core import parallel_state
from deepfold.core.utils import divide


def split_tensor(
    tensor: torch.Tensor,
    num_partitions: int,
    dim: int = -1,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """
    Split a tensor along its last dimension.

    Arguments:
        tensor (torch.Tensor): input tensor.
        num_partitions (int): number of partitions to split the tensor.
        dim (int): the dimension to be splitted.
        contiguous_split_chunks: If True, make each chunk contiguous in memory.

    Returns:
        A list of tensors.
    """

    # Get the size and dimension
    ndim = tensor.dim()
    dim = dim + ndim if dim < 0 else dim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"Dim {dim} is out of bound")
    dim_size = divide(tensor.size()[dim], num_partitions)

    # Split
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def split_tensor_into_1d_equal_chunks(tensor: torch.Tensor, new_buffer: bool = False) -> torch.Tensor:
    """
    Break a tensor into equal 1D chunks acorss model parallel ranks.

    Returns a Tensor or View.

    Arguments:
        tensor (torch.Tensor): the tensor to split.
        new_buffer (bool): If True, returns a new Tensor.
                           If False, return a View into the existing Tensor.
    """

    partition_size = torch.numel(tensor) // parallel_state.get_model_parallel_world_size()
    start_index = partition_size * parallel_state.get_model_parallel_rank()
    end_index = start_index + partition_size

    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]

    return data


def gather_split_1d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Opposite of split_tensor_into_1d_equal_chunks.
    """

    numel_gathered = torch.numel(tensor) * parallel_state.get_model_parallel_world_size()
    gathered = torch.empty(
        numel_gathered,
        dtype=tensor.dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    torch.distributed.all_gather_into_tensor(
        gathered,
        tensor,
        group=parallel_state.get_model_parallel_group(),
    )

    return gathered
