import logging
from typing import List, Optional

import torch
from torch.distributed._functional_collectives import all_gather_tensor, reduce_scatter_tensor

import deepfold.distributed._backend as cc
from deepfold.distributed.device_mesh import DeviceMesh

__all__ = [
    "all_gather_tensor",
    "reduce_scatter_tensor",
    "mesh_scatter",
    "mesh_broadcast",
    "mesh_all_to_all",
]


logger = logging.getLogger(__name__)


def mesh_scatter(
    output: torch.Tensor,
    scatter_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[cc.Work]:
    """
    Scatter a list of tensors to a device mesh dimension. (One-to-all).
    Use the first rank of the mesh dimension as the source by default.

    Args:
        output: torch.Tensor
            The tensor to receive the scattered list.
        scatter_list: List[torch.Torch]
            List of tensors to be scattered.
        mesh_dim: int, optional
            Mesh dimension on which to be scattered.

    Returns:
        A `Work` object.
    """
    if output.is_meta:
        return None

    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, cc.ProcessGroup)

    # Be global rank
    src_for_dim = 0
    if dim_group is not cc.GroupMember.WORLD:
        src_for_dim = cc.get_global_rank(dim_group, 0)

    if src_for_dim == cc.get_rank():
        fut = cc.scatter(
            output,
            scatter_list=scatter_list,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )
    else:
        fut = cc.scatter(
            output,
            scatter_list=None,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )

    return fut


def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[cc.Work]:
    """
    Broadcast the tensor to a device mesh dimension.
    Use the first rank of the mesh dimension as the source by default.

    Args:
        tensor: torch.Tensor
            Tensor to broadcast.
        mesh_dim: int, optional
            Mesh dimension on which to be broadcasted.

    Returns:
        A `Work` object.
    """
    if tensor.is_meta:
        return None

    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, cc.ProcessGroup)

    # Be global rank
    src_for_dim = 0
    if dim_group is not cc.GroupMember.WORLD:
        src_for_dim = cc.get_global_rank(dim_group, 0)

    return cc.broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)


def mesh_all_to_all(
    output_tensor_list: List[torch.Tensor],
    intput_tensor_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[cc.Work]:
    """
    Args:
        output_tensor_list: List[torch.Tensor]
            List of tensors to be gathered one per rank.
        intput_tensor_list: List[torch.Tensor]
            List of tensors to scatter one per rank.
        mesh_dim: int, optional
            Mesh dimension on which to be broadcasted.

    Returns:
        A `Work` object.
    """
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, cc.ProcessGroup)

    work = None
    # No dist.all_to_all on 'gloo'
    if mesh.device_type == "cpu":
        logger.warning("Gloo does not support all_to_all")
        dim_group_size = cc.get_world_size(dim_group)
        for i in range(dim_group_size):
            # Be global rank
            src_for_dim = i
            if dim_group is not cc.GroupMember.WORLD:
                src_for_dim = cc.get_global_rank(dim_group, i)

            work = cc.scatter(
                output_tensor_list[i],
                intput_tensor_list if mesh.get_rank() == src_for_dim else [],
                group=dim_group,
                src=src_for_dim,
                async_op=async_op,
            )
    else:
        work = cc.all_to_all(
            output_tensor_list,
            intput_tensor_list,
            dim_group,
            async_op=async_op,
        )

    return work


# TODO: Cost estimation
