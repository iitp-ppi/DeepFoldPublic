from typing import List, Optional

import torch
from torch.distributed import GroupMember, Work, broadcast, get_global_rank, get_rank, scatter
from torch.distributed.distributed_c10d import ProcessGroup

import deepfold.distributed.tensor.placement_types as placement_types
from deepfold.distributed.tensor.device_mesh import DeviceMesh, _mesh_resources


def mesh_scatter(
    output: torch.Tensor,
    scatter_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    Scatter a list of tensors to a device mesh dimension. We by default use the first rank of the mesh dimension
    as the source, i.e., for a 2-dimensional mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank 2 to rank 2/3.

    Args:
        output: torch.Tensor
            the tensor to receive the scattered list.
        scatter_list: List[torch.Tensor]
            the tensor list to be scattered.
        mesh_dim: int, optional
            indicate which mesh dimension we want to scatter on, the first rank on the mesh dimension by default.
    Returns:
        A `Work` object.
    """
    if output.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # src need to be global rank
    src_for_dim = 0

    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)

    if src_for_dim == get_rank():
        work = scatter(output, scatter_list=scatter_list, src=src_for_dim, group=dim_group, async_op=async_op)
    else:
        work = scatter(output, scatter_list=None, src=src_for_dim, group=dim_group, async_op=async_op)

    return work


def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    Broadcast the tensor to a device mesh dimension. We by default use the first rank of the mesh dimension as
    the source.

    Args:
        tensor: torch.Tensor
            tensor to broadcast
        mesh_dim: int, optional
            indicate which mesh dimension we want to scatter on, the first rank on the mesh dimension by default.
    Returns:
        A `work` object
    """
    if tensor.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # src need to be global rank
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)

    return broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)


# TODO: mesh_all_to_all

# TODO: spec_to_bytes

# TODO: redistribute_cost
