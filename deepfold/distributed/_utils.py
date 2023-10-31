from typing import List, Sequence, Tuple, Union, cast

import torch

from deepfold.distributed.device_mesh import DeviceMesh
from deepfold.distributed.placement_types import Placement, Replicate, Shard, _Partial

ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]

__all__ = [
    "compute_local_shape",
    "compute_local_shape_and_global_offset",
    "compute_global_tensor_info",
]


def compute_local_shape(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current coordinate of the mesh.
    """
    my_coord = mesh.get_coordinate()

    if my_coord is None:
        # If rank not in the mesh, return empty shape
        return ()
    else:
        local_shape = list(global_shape)
        ndim = len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert shard_dim < ndim, f"Sharding dim {shard_dim} greater than tensor ndim {ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coord[idx],
                )
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size

        return tuple(local_shape)


def compute_local_shape_and_global_offset(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor of a DTensor on its current global rank.
    """
    my_coord = mesh.get_coordinate()

    if my_coord is None:
        # If rank not in the mesh, return empty offset
        return ((), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coord[idx],
                    return_offset=True,
                )
                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # On a given dim, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                # it means that this dimension has been already sharded in previous placement.
                # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                # Instead, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

        return tuple(local_shape), tuple(global_offset)


def compute_global_tensor_info(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Tuple[List[int], List[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplied by `world_size` per sharding dim.
    The local stride is multiplited by `world_size` per sharding dim, as ...

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8) where
    the DTensor placements are [Shard(2)] and world_size is 2, then the global size is (4, 8, 4)
    and the stirde is (16 * 2, 1, 8).

    Args:
        tensor: torch.Tensor
            Local tensor which DTensor will be constructed from.
        mesh: DeviceMesh
            Device mesh for the DTensor.
        placements: Sequence[Placement]
            Layout of the DTensor on the mesh.

    Returns:
        tensor_shape:
            A list of `int` which specifies the size of DTensor which build on the local tensor.
        tensor_stride:
            A list of `int` which specifies the stride of DTensor.
    """
    tensor_shape = list(tensor.size())
    tensor_stride = list(tensor.stride())

    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if placement.is_shard():
            shard_placement = cast(Shard, placement)
            if shard_placement.dim < 0:
                # Normalize shard dim to be positive
                shard_placement.dim += len(tensor_shape)
            shard_dim = shard_placement.dim

            local_dim_size = tensor_shape[shard_dim]
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size

            # Recover tensor stride by modifying the stride that larger than
            # the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # Rescale the stride by the shard size
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size

        elif not isinstance(placement, (Replicate, _Partial)):
            raise RuntimeError(f"Placement type {type(placement)} not supported")

    return tensor_shape, tensor_stride
