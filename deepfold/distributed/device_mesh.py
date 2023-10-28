import logging
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

import deepfold.distributed.comm as _comm
from deepfold.distributed import backend as _backend


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _get_device_handle(device_type: str = "cuda"):
    return getattr(torch, device_type, None)


class MeshEnv:
    @staticmethod
    def num_devices_per_host(device_type: str) -> int:
        return _get_device_handle(device_type).device_count()

    @staticmethod
    def num_hosts(device_type: str) -> int:
        return _backend.get_world_size() // MeshEnv.num_devices_per_host(device_type)


class DeviceMesh:
    """
    DeviceMesh represents a mesh of devices, where layout of devices could be represented as a
    n-dimensional array, and each value of the n-dimensional array is the global id of the default
    process group ranks.
    """

    device_type: str
    mesh: torch.Tensor
    mesh_dim_names: Optional[Tuple[str, ...]]

    def __init__(
        self,
        device_type: str,
        mesh: Union[torch.Tensor, "ArrayLike"],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        _init_process_groups: bool = True,
        _validate_mesh: bool = True,
    ) -> None:
        self.device_type = device_type
        self.mesh = mesh.detach() if isinstance(mesh, torch.Tensor) else torch.tensor(mesh, dtype=torch.int)
        self.mesh_dim_names = mesh_dim_names

        self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
        self._hash = hash((self._flatten_mesh_list, self.mesh.shape))

        self._get_or_create_default_group()
        if _init_process_groups:
            self._init_process_groups(_validate_mesh)

    def _get_or_create_default_group(self):
        default_initialized = _backend.is_initialized()
        if not default_initialized:
            _backend.init_process_group()

        world_size = _backend.get_world_size()
        if self.mesh.numel() > world_size:
            raise f"Mesh should not be bigger than default world size, but {self.mesh.numel()}"

        device_handle = _get_device_handle()

        if not default_initialized and device_handle:
            num_devices_per_host = MeshEnv.num_devices_per_host()
            if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found "
                    f"{world_size} ranks and {num_devices_per_host} {self.device_type} devices"
                )
            device_handle.set_device(_backend.get_rank())

        # Calculate the coordinates of the current global rank on the mesh
        rank_coords = (self.mesh == _backend.get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinates_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None

        return _backend._get_default_group()

    def _init_process_group(self, _validate_mesh):
        if _validate_mesh:
            self._validate_mesh()

        # Group tag associated with each mesh dimension
        # Each mesh dimension should have one subgroup per rank
        dim_group_infos: List[Tuple[str, List[int]]] = []

        if self.mesh.ndim == 1 and self.mesh.numel() == _backend.get_world_size():
            # If the mesh is the same as world_pg, just append the default pg to the first dim groups
            dim_group_infos.append(
                _backend._get_group_tag(_backend._get_default_group()), list(range(_backend.get_world_size()))
            )
        else:
            # Create sub-pgs
            for dim in range(self.mesh.ndim):
                # Swap the current dim to the last dim then reshape to flatten out other dims
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                # Multi-dimensional mesh, create subgroups
                # by looping over the pg_ranks for each dim and append the gruops
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # Call new_group regardless of the current rank in the pg or not
                    # It is required that all ranks participate in subgroup construction
                    dim_group = _backend.new_group(ranks=subgroup_ranks)
                    # Only add to dim_groups if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_group_infos) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process group, "
                                f"but got {self.get_rank()} in {subgroup_ranks}"
                            )
                        dim_group_infos.append((_backend._get_group_tag(dim_group), subgroup_ranks))

        self._dim_group_infos = dim_group_infos

    def __repr__(self) -> str:
        return (
            f"DeviceMesh(device_type='{self.device_type}'\n"
            f"mesh={self.mesh.tolist()})\n"
            f"mesh_dim_names={self.mesh_dim_names}"
        )

    def __hash__(self) -> int:
        self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self.mesh) == id(other.mesh):
            return True
        return self.mesh.shape == other.mesh.shape and self._flatten_mesh_list == other._flatten_mesh_list

    def get_dim_groups(
        self, mesh_dim: Optional[int] = None
    ) -> Union[_backend.ProcessGroup, List[_backend.ProcessGroup]]:
        if not hasattr(self, "_dim_group_infos"):
            raise RuntimeError("DeviceMesh process groups not initialized")
        if mesh_dim is not None:
            return _backend._find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
        else:
            dim_groups = []
            for mesh_dim in range(self.mesh.ndim):
                dim_groups.append(_backend._find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim]))
            return dim_groups

    def size(self, dim: Optional[int] = None) -> int:
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    def get_rank(self) -> int:
        return _backend.get_rank()

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the coordinate of the device in the mesh.
        If this rank is not part of the mesh, return None.
        """
        return self._coordinates_on_dim if self._coordinates_on_dimm else None

    def _validate_mesh(self):
        # Check mesh tensor validity
        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(f"DeviceMesh cannot have duplicated values, buf found {self.mesh.tolist()}")

        # Validate that all calling ranks has the same mesh
        self_mesh = self.mesh.to(self.device_type).contiguous()
        mesh_tensor = _comm.all_gather_tensor(self_mesh, gather_dim=0, group=_backend._get_default_group())
        mesh_tensor_chunked = torch.chunk(mesh_tensor, _backend.get_world_size())
        for other_rank, other_mesh in enumerate(mesh_tensor_chunked):
            if not torch.equal(self_mesh, other_mesh):
                raise RuntimeError(
                    f"DeviceMesh initialization does not allow different mesh arguments: "
                    f"rank {_backend.get_rank()} has mesh {self_mesh} while rank {other_rank} "
                    f"has mesh {other_mesh}"
                )


def init_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
) -> "DeviceMesh":
    """
    Initialize a `DeviceMesh` based on parameters.
    """
    if mesh_dim_names is not None:
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError(
                f"Each mesh dimension name must be unique, but found repeated name in " f"{mesh_dim_names}"
            )
        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(f"`mesh_shape` and `mesh_dim_names` should have same length")

    mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
    device_mesh = DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)

    return device_mesh
