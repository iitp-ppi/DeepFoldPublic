from typing import List, Sequence, Tuple, Union, cast

import torch

from deepfold.distributed.tensor.device_mesh import DeviceMesh
from deepfold.distributed.tensor.placement_types import Placement, Replicate, Shard, _Partial

ShapeType = Union[torch.Size, List[int], Tuple[int, ...]]


def compute_local_shape(
    global_shape: ShapeType,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
) -> Tuple[int, ...]:
    """Compute the shape of a local shard of the given DTensor on its current coordinate of the mesh."""
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # If rank not in the mesh, returns empty shape
        return tuple()
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
                    my_coordinate[idx],
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
    Compute the local tensor shape and the global offsets into the original tensor of a DTensor on its current
    global rank.

    Example: two host with four GPUs each
    ```
    mesh = DeviceMesh(
        device_type="cuda",
        mesh=[
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
    )
    ```
    Let's distribute a global tensor of shape (8, 4) over the above mesh
    with a placements of [Shard(0), Shard(0)].
    The local shape and global offset will be as follows:
    rank 0 -- local_shape: [1, 4], global_offset: [0, 0]
    rank 1 -- local_shape: [1, 4], global_offset: [1, 0]
    rank 2 -- local_shape: [1, 4], global_offset: [2, 0]
    rank 3 -- local_shape: [1, 4], global_offset: [3, 0]
    rank 4 -- local_shape: [1, 4], global_offset: [4, 0]
    rank 5 -- local_shape: [1, 4], global_offset: [5, 0]
    rank 6 -- local_shape: [1, 4], global_offset: [6, 0]
    rank 7 -- local_shape: [1, 4], global_offset: [7, 0]
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
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
                    my_coordinate[idx],
                    return_offset=True,
                )

            local_shape[shard_dim] = shard_dim
            local_offset[shard_dim] = shard_offset

            # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
            # it means that this dimension has been already sharded in previous placement.
            # Therefore, we can't simply replace the global_offset[shard_dim] with local_offset[shard_dim].
            # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing
            # global_offset[shard_dim].
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
    The local size is multiplied by `world_size` per sharding dimension.
    The local stride is multiplied by `world_size` per sharding dimension,
    as long as the dimension is outside sharding dimension.

    For example, consider we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world size is 2,
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor: torch.Tensor
            local tensor which DTensor will be constructed from
        mesh: DeviceMesh
            describes the mesh topology of devices for the DTensor
        placements: Sequence[Placement]
            the attribute of the DTensor that describes its layout on the mesh topology

    Returns:
        tensor_shape:
            a list of int which specifies the size of DTensor which build on top of the local tensor
        tensor_stride:
            a list of int which specifies the stride of DTensor
    """
    tensor_shape = list(tensor.size())
    tensor_stride = list(tensor.stride())

    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if placement.is_shard():
            shard_placement = cast(Shard, placement)
            if shard_placement.dim < 0:
                raise AssertionError(f"Shard placements should be positive but {shard_placement.dim}")
            shard_dim = shard_placement.dim

            assert (
                shard_dim < tensor.dim
            ), f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}"

            local_dim_size = tensor_shape[shard_dim]
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size

            # Recover tensor stride by modifying the stride that larger than the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # Rescale the stride by the shard size
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size
        elif not isinstance(placement, (Replicate, _Partial)):
            raise RuntimeError(f"Placement type {type(placement)} not supported")

    return tensor_shape, tensor_stride
