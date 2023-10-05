import logging
import types
from math import prod
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch

from deepfold.distributed.core import (
    ProcessGroup,
    all_gather_tensor,
    get_default_group,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    new_group,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _get_device_handle(device_type: str = "cuda") -> types.ModuleType:
    return getattr(torch, device_type, None) if device_type != "cpu" else None


class DeviceMesh:

    """
    `DeviceMesh` represents a mesh of devices, where layout of devices is a n-dimensional array, and each value of
    of the n-dimensional array is the global id of the default process group ranks.

    `DeviceMesh` could be used to describe the layout of devices across the cluster, and serve as a proxy for
    communication between devices within the cluster.

    Todo:
        We use the default ProcessGroup in this `DeviceMesh` class to implement proper communications.
        We will add collective wrappers in this class.
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
        """
        Args:
            device_type:
                device type of the mesh.
            mesh:
                could be a multi-dimensional array or an integer tensor that describes the layout of devices,
                the ids are global ids of the default process group.
        """

        self.device_type = device_type
        self.mesh = mesh.detach() if isinstance(mesh, torch.Tensor) else torch.tensor(mesh, dtype=torch.int)
        if mesh_dim_names is None:
            mesh_dim_names = [str(i) for i in range(self.mesh.ndim)]
        self.mesh_dim_names = mesh_dim_names

        self._get_or_create_default_group()
        if _init_process_groups:
            self._init_process_groups(_validate_mesh)

    def _get_or_create_default_group(self) -> ProcessGroup:
        # Initialization check
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()

        # Size check
        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"DeviceMesh should not be bigger than default world size, but found {self.mesh.numel()} ranks"
            )

        device_handle = _get_device_handle(self.device_type)
        if not default_initialized and device_handle:
            num_devices_per_host = device_handle.device_count()
            if world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardward, but found {world_size} ranks and"
                    f" {num_devices_per_host} {self.device_type} devices"
                )
            device_handle.set_device(get_rank() % num_devices_per_host)

        # Calculate the coordinates of the current global rank on the mesh
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None

        return get_default_group()

    def _validate_mesh(self):
        # Check mesh tensor validity
        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(f"DeviceMesh cannot have duplicate values, but found {self.mesh.tolist()}")

        # Validate all ranks have same mesh `mesh` argument
        self_mesh = self.mesh.to(self.device_type)
        mesh_tensor = all_gather_tensor(self_mesh, gather_dim=0, group=get_default_group())
        # mesh_tensor_chunked = torch.chunk(mesh_tensor, get_world_size())
        for other_rank, other_mesh in enumerate(mesh_tensor):
            if not torch.equal(self_mesh, other_mesh):
                raise RuntimeError(
                    f"DeviceMesh initialization does not allow different mesh argument:"
                    f" rank {get_rank()} has mesh {self_mesh} while rank {other_rank}"
                    f" has mesh {other_mesh}"
                )

    def _init_process_groups(self, _validate_mesh: bool):
        if _validate_mesh:
            self._validate_mesh()

        # Group ranks associated with each mesh dimension
        # Each mesh dimension should have one sub-group per rank
        dim_groups: List[Tuple[str, ProcessGroup]] = []

        if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
            # If the mesh is the same as world_pg, we just append the default pg to the first dim gruops
            dim_groups.append((self.mesh_dim_names[0], list(range(get_world_size()))))
        else:
            # Create sub-pgs base on the mesh argument specified
            for dim in range(self.mesh.ndim):
                # Swap the current dim to the last dim then reshape to flatten out other dims
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                # For a multi-dimensional mesh, create subgroups for each dim
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # Only add to `dim_groups` if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_groups) > dim:
                            raise RuntimeError(
                                f"Each mesh dimension should get only one process group, but {self.get_rank()}"
                                f" in {subgroup_ranks}"
                            )
                        dim_group = new_group(ranks=subgroup_ranks)
                        dim_groups.append((self.mesh_dim_names[dim], dim_group))
        self._dim_groups = dim_groups

    def __repr__(self) -> str:
        return f"DeviceMesh({self.mesh.tolist()})"

    def __hash__(self) -> int:
        return hash((self.mesh, id(self)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self) == id(other):
            return True
        return self.mesh.equal(other.mesh)

    def get_dim_groups(self, mesh_dim: Optional[int] = None) -> Union[ProcessGroup, List[ProcessGroup]]:
        """
        Get dimension a process group correspond to `mesh_dim`.
        If `mesh_dim` is `None` return the list of all process groups.
        """
        if not hasattr(self, "_dim_groups"):
            raise RuntimeError("DeviceMesh process groups not initialized")
        if mesh_dim is not None:
            return self._dim_groups[mesh_dim][1]
        else:
            return [pg for _, pg in self._dim_groups]

    def size(self, dim: Optional[int] = None) -> int:
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    def ndim(self) -> int:
        return self.mesh.ndim

    def get_rank(self) -> int:
        return get_rank()

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all dimensions of the mesh.
        If this rank is not part of the mesh, return `None`.
        """
        return self._coordinate_on_dim if self._coordinate_on_dim else None


def init_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Optional[Sequence[str]] = None,
) -> DeviceMesh:
    """
    Initializes a `DeviceMesh` based on arguments.
    """
    if mesh_dim_names is not None:
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError("Each `mesh_dim_names` must be unique")
        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError("`mesh_shape` and `mesh_dim_names` must have same length")

    mesh = torch.arange(prod(mesh_shape)).view(mesh_shape)
    device_mesh = DeviceMesh(device_type=device_type, mesh=mesh, mesh_dim_names=mesh_dim_names)

    return device_mesh
