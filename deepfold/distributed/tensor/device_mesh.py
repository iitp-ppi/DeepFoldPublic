# Copyright 2023 DeepFold Team


import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from torch.distributed import get_rank, get_world_size, init_process_group, is_initialized, new_group
from torch.distributed.distributed_c10d import (
    ProcessGroup,
    _find_pg_by_ranks_and_tag,
    _get_default_group,
    _get_group_tag,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class _MeshEnv:
    def __init__(self) -> None:
        self.mesh_stack = List[DeviceMesh] = []
        self.child_to_parent_mapping: Dict[DeviceMesh, DeviceMesh] = {}

    def get_current_mesh(self) -> "DeviceMesh":
        if len(self.mesh_stack) == 0:
            raise RuntimeError("No device mesh")
        return self.mesh_stack[-1]

    def create_child_mesh(self, device_mesh: "DeviceMesh", mesh_dim: int, mesh_dim_name: str) -> "DeviceMesh":
        # Swap the current dim to the last dim then reshape to flatten out other dims,
        # so that we can just extract the list of ranks which contains current rank.
        pass  # TODO: Implement

    def get_parent_mesh(self, device_mesh: "DeviceMesh") -> Optional["DeviceMesh"]:
        """
        Return the parent mesh of the device mesh passed in.
        """
        return self.child_to_parent_mapping.get(device_mesh, None)

    def get_parent_mesh_dim(self, device_mesh: "DeviceMesh") -> Optional[int]:
        """
        Return the index of the mesh dim in the parent mesh.
        The device mesh passed in needs to be sliced out from a parent mesh.
        """

    @staticmethod
    def num_devices_per_host(device_type: str) -> int:
        return _get_device_handle(device_type).device_count()

    @staticmethod
    def num_hosts(device_type: str) -> int:
        # Assume homogeneous hardware for now.
        return get_world_size() // _MeshEnv.num_devices_per_host(device_type)


_mesh_resources: _MeshEnv = _MeshEnv()


def _get_device_handle(device_type: str = "cuda"):
    """
    Get the module corresponding to the device_type.
    """
    return getattr(torch, device_type, None)


class DeviceMesh:
    """
    `DeviceMesh` represents a mesh of devices, where layout of devices could be represented as a n-dimensional array,
    and each value of the array is the global id of the default process group ranks.

    `DeviceMesh` could be used to describe the layout of devices across the cluster, and serves a proxy for
    communication among the device in the cluster.

    We use the default process group in the device mesh class to implement proper communications.

    Note that we also add collective wrappers in this class. This is used to decouple detailed communication
    backend with the underlying distributed tensor implementation.

    `DeviceMesh` can be used as a context manager.

    `DeviceMesh` follows SPMD programming model, which means the same PyTorch python program is running on all ranks
    in the cluster. Therefore, user need to make sure the `mesh` array should be identical across all ranks.
    Inconsistent `mesh` will lead undifined behaviours. (Mostly slient hang.)
    """

    def __init__(
        self,
        device_type: str,
        mesh: Union[torch.Tensor, "ArrayLike"],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        _init_process_groups: bool = True,
    ) -> None:
        """
        Args:
            device_type: str
                device type of the mesh. Currently supports: cpu and cuda.
            mesh: ArrayLike
                could be a multi-dimensional array or an integer tensor that describes the layout of devices.
                The ids are global ids of the default process group.
        """
        self.device_type = device_type
        self.mesh = mesh.detach() if isinstance(mesh, torch.Tensor) else torch.tensor(mesh, dtype=torch.int)
        self.mesh_dim_names = mesh_dim_names

        self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
        self._hash = hash((self._flatten_mesh_list, self.mesh.shape))

        self._get_or_create_default_group()
        if _init_process_groups:
            self._init_process_groups()

    def __get_or_create_default_group(self) -> ProcessGroup:
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()

        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be larger than default world size, but found {self.mesh.numel()} ranks"
            )

        device_handle = _get_device_handle(self.device_type)

        if not default_initialized and device_handle:
            # Automatically set the current device base on number of devices available in each host
            num_devices_per_host = device_handle.device_count()
            if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found"
                    f" {world_size} ranks and {num_devices_per_host} {self.device_type} devices"
                )
            device_handle.set_device(get_rank() % num_devices_per_host)

        #  Calculate the coordinates of the current global rank on the mesh
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None

        return _get_default_group()

    def _init_process_groups(self) -> None:
        # Group tag/ranks associated with each mesh dimension
        # Each mesh dimension should have one sub-group per rank
        dim_group_infos: List[Tuple[str, List[int]]] = []

        if self.mesh.ndim == 1 and self.mesh.numel() == get_world_size():
            # If the mesh is the same as WORLD, we just append the default process group to the first dimension groups
            dim_group_infos.append(_get_group_tag(_get_default_group()), list(range(get_world_size())))
        else:
            # Create sub-process groups base on the mesh argument specified
            for dim in range(self.mesh.ndim):
                # Swap the current dimension to the last dimension
                # then reshape to flatten out other dimensions
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                # Multi-dimensional mesh
                # Create subgroups by lopping over the process group ranks for each dimension and append the groups
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # It's required that all ranks participate in subgroup construction
                    dim_group = new_group(ranks=subgroup_ranks)
                    # Only add to dimension group if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_group_infos) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process gruop,"
                                f" but got {self.get_rank()} in {subgroup_ranks}"
                            )
                        dim_group_infos.append((_get_group_tag(dim_group), subgroup_ranks))
        self._dim_group_infos = dim_group_infos

    def __enter__(self) -> "DeviceMesh":
        # Set this mesh as the current mesh in mesh env
        _mesh_resources.mesh_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # Pop this mesh from mesh env
        _mesh_resources.mesh_stack.pop()

    def __repr__(self) -> str:
        return f"DeviceMesh({self.mesh.tolist()})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self.mesh) == id(other.mesh):
            return True
        return self.mesh.shape == other.mesh.shape and self._flatten_mesh_list == other._flatten_mesh_list

    def __getitem__(self, mesh_dim_name: str) -> "DeviceMesh":
        """
        Slice the current device mesh based on the dimension name given to create child device mesh.

        Args:
            mesh_dim_name: str
                the name of the mesh dimension of the parent `DeviceMesh` to create a child `DeviceMesh`.
        Returns:
            A `DeviceMesh` object.

        Example:
        Two hosts with 4 GPUs each.
        ```
        mesh = DeviceMesh(
            device_type="cuda",
            mesh = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ],
            mesh_dim_names=["dp", "tp"]
        )
        ```
        Call mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child mesh: [0, 1, 2, 3].
        Call mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child mesh: [4, 5, 6, 7].
        Call mesh["dp"] on rank 0, 4 would return a 1D child mesh: [0, 4].
        Call mesh["dp"] on rank 2, 6 would return a 1D child mesh: [2, 6].
        """
        if self.mesh.ndim <= 1:
            raise RuntimeError(f"Cannot slice a DeviceMesh with {self.mesh.ndim} dimension.")
        if self.mesh_dim_names is None:
            raise KeyError(f"No `mesh_dim_names` found")
        if mesh_dim_name not in self.mesh_dim_names:
            raise KeyError(
                f"Mesh dimension '{mesh_dim_name}' does not exist."
                f" Available mesh dimensions are: {self.mesh_dim_names}"
            )
        mesh_dim = self.mesh_dim_names.index(mesh_dim_name)
        sub_mesh = _mesh_resources.create_child_mesh(self, mesh_dim, mesh_dim_name)

        return sub_mesh

    def get_dim_groups(self, mesh_dim: Optional[int] = None) -> Union[ProcessGroup, List[ProcessGroup]]:
        if not hasattr(self, "_dim_group_infos"):
            raise RuntimeError("DeviceMesh process groups not initialized!")
        if mesh_dim is not None:
            return _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
        else:
            dim_groups = []
            for mesh_dim in range(self.mesh.ndim):
                dim_groups.append(_find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim]))
            return dim_groups

    def size(self, dim: Optional[int] = None) -> int:
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.mesh.shape)

    def get_rank(self) -> int:
        return get_rank()

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all dimensions of the mesh.
        If this rank is not part of the mesh, return None.
        """
        return self._coordinate_on_dim if self._coordinate_on_dim else None


def init_device_mesh(
    device_type: str, mesh_shape: Tuple[int, ...], *, mesh_dim_names: Optional[Tuple[str, ...]] = None
) -> DeviceMesh:
    """
    Initialize a `DeviceMesh` based on `device_type`, `mesh_shape` and `mesh_dim_names` parameters.
    This creates a device mesh with a mesh layout of n-dimensional array, `n` being the `len(mesh_shape)`
    and i-th dimension being in size `mesh_shape[i]`.
    If `mesh_dim_names` is provided, each dimension is labeled as `mesh_dim_names[i]`.

    Notes:
        `init_device_mesh` follows SPMD programming model, which means the same PyTorch program is running
        on all processes/ranks in the cluster. Therefore, users need to make sure the `mesh_shape` tuple
        should be identical across all ranks.

    Args:
        device_type: str
            device type of the mesh. Currently supports: cpu and cuda.
        mesh_shape: Tuple[int]
            defines the dimension of the multi-dimensional array that describes the layout of devices.
        mesh_dim_names: Tuple[str], optional
            are asseinged to each dimension of the mesh.
            Its length must match the length of `mesh_shape`.
            Each string in `mesh_dim_names` must be unique.

    Returns:
        A `DeviceMesh` object.
    """
    if mesh_dim_names is not None:
        # Check uniqueness
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError(f"Each `mesh_dim_name` must be unique. Found repeated in {mesh_dim_names}")

        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(
                f"`mesh_shape` (len={len(mesh_shape)}) and `mesh_dim_names` (len={len(mesh_dim_names)}) should have same length."
            )

    mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
    device_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
    )

    return device_mesh
