from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Tuple, cast

import torch
from torch.distributed import all_gather, all_reduce, reduce_scatter
from torch.distributed.distributed_c10d import ReduceOp

from deepfold.distributed.tensor.collective_utils import mesh_broadcast, mesh_scatter
from deepfold.distributed.tensor.device_mesh import DeviceMesh


class Placement:
    def is_shard(self, dim: Optional[int] = None) -> bool:
        is_shard_instance = isinstance(self, Shard)
        if dim is not None and is_shard_instance:
            return cast(Shard, self).dim == dim
        else:
            return is_shard_instance

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, _Partial)


@dataclass(frozen=True)
class Shard(Placement):
    dim: int

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Use `torch.chunk` to split a tensor into `num_chunks` shards along the shard placement dimension, and
        return a list of shards with their pad sizes.

        Args:
            tensor: torch.Tensor
                a tensor to split
            num_chunks: int
                number of chunks to return
            with_padding: bool, optional
                when `True`, we pad the tensor on the last few ranks before calling the collectives. This is
                because collectives usually require eqaul size tensor inputs
        """
        assert self.dim <= tensor.ndim, f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
        assert tensor.size(self.dim) > 0, f"Tensor size along dim={self.dim} is 0. Nothing to be sharded"

        # Chunk tensor over dimension `dim` into n slices with padding if necessary
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim))
        # Compute the chunk size inline with `torch.chunk`
        full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk for `self.dim`
        chunk_sizes = [tensor_list[idx].size(self.dim) if idx < len(tensor_list) else 0 for idx in range(num_chunks)]
        # Compute pad size on each chunk
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]

        # Reuse tensor to fill empty chunk with empty tensor
        num_empty_tensors = num_chunks - len(tensor_list)
        tensor_size = list(tensor_list[0].size())
        tensor_size = [size if idx != self.dim else 0 for idx, size in enumerate(tensor_size)]
        tensor = tensor.new_zeros(tensor_size)
        for _ in range(num_empty_tensors):
            tensor_list.append(tensor)

        if with_padding or contiguous:
            shard_list = []
            for shard, pad_size in zip(tensor_list, pad_sizes):
                # Fill the empty tensor with zeros with padding
                if with_padding and pad_size > 0:
                    shard = self._pad_tensor(shard, pad_size)
                shard = shard.contiguous() if contiguous else shard
                shard_list.append(shard)
            return shard_list, pad_sizes
        else:
            return tensor_list, pad_sizes

    def _pad_tensor(
        self,
        tensor: torch.Tensor,
        pad_size: int,
    ) -> torch.Tensor:
        pad = [0, 0] * (tensor.ndim - self.dim)
        pad[-1] = pad_size
        return torch.nn.functional.pad(tensor, pad)

    def _unpad_tensor(self, tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        return tensor.narrow(self.dim, start=0, length=tensor.size(self.dim) - pad_size)

    def _local_shard_size_on_dim(
        self,
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
    ) -> Tuple[int, int]:
        """Returns the local shard size and offset on a given tensor dim."""
        assert (
            size_on_dim >= num_chunks
        ), f"Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}"

        # Compute the chunk size inline with `torch.chunk`
        full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk on the dimension
        chunk_sizes = [
            max(
                min(size_on_dim, full_chunk_size * (idx + 1)) - full_chunk_size * idx,
                0,
            )
            for idx in range(num_chunks)
        ]
        local_shard_size = chunk_sizes[rank]

        local_offset_on_dim = -1
        if return_offset:
            # Return global tensor dim size of current dimension
            # if for empty shard to represent the end of the corresponding tensor dim
            local_offset_on_dim = sum(chunk_sizes[:rank])

        return local_shard_size, local_offset_on_dim

    def _shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """Shard and scatter a tensor on a mesh dimension (use coordinate 0 on the mesh dimension as source)."""
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)

        if my_coordinate is None:
            # If rank is not part of mesh, return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        scatter_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)

        output = torch.empty_like(scatter_list[my_coordinate[mesh_dim]])
        mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)

        # Only unpad if the local tensor was padded on the dimension
        pad_size = pad_sizes[my_coordinate[mesh_dim]]
        if pad_size > 0:
            output = self._unpad_tensor(output, pad_size)

        return output

    def _reduce_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        reduce_op: ReduceOp.RedOpType,
        mesh_dim: int,
    ) -> torch.Tensor:
        """Reduce and scatter a tensor on a mesh dimension."""
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)

        if my_coordinate is None:
            # If rank is not part of mesh, we simply return local tensor,
            # which should be an empty tensor
            return tensor

        is_padded = tensor.size(self.dim) % num_chunks != 0

        if is_padded:
            scattered_list, pad_sizes = self._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
        else:
            scattered_list = [t.contiguous() for t in torch.chunk(tensor, num_chunks, dim=self.dim)]

        # Prepare an output tensor
        tensor = scattered_list[my_coordinate[mesh_dim]]
        tensor = tensor.new_empty(tensor.size())

        # TODO: Check
        reduce_scatter(
            tensor,
            scattered_list,
            op=reduce_op,
            group=mesh.get_dim_groups(mesh_dim),
            async_op=False,
        )

        if is_padded:
            tensor = self._unpad_tensor(tensor, pad_sizes[my_coordinate[mesh_dim]])

        return tensor

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        size: torch.Size,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        All-gather all shards and return a tensor but is replicated on the previously sharded mesh dimension.
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(dim=mesh_dim)

        if my_coordinate is None:
            # If rank is not part of mesh, simply return local_tensor
            return local_tensor

        # Check if it need to pad input tensor before all_gather
        full_chunk_size = (size[self.dim] + num_chunks - 1) // num_chunks
        chunk_sizes = [
            max(min(size[self.dim], full_chunk_size * (idx + 1)) - full_chunk_size * idx, 0)
            for idx in range(num_chunks)
        ]
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]
        is_padded = size[self.dim] % num_chunks != 0

        pad_size = pad_sizes[my_coordinate[mesh_dim]]
        if pad_size > 0:
            local_tensor = self._pad_tensor(local_tensor, pad_size)
        local_tensor = local_tensor.contiguous()

        # TODO: Check
        tensor_list = [local_tensor.new_empty(local_tensor.siez()) for _ in range(num_chunks)]
        all_gather(
            tensor_list,
            local_tensor,
            group=mesh.get_dim_groups(mesh_dim),
            async_op=False,
        )
        tensor = torch.cat(tensor_list, dim=mesh_dim)

        # Unpad the tensor if the input tensor was padded
        if is_padded:
            full_pad_size = sum(pad_sizes)
            tensor = self._unpad_tensor(tensor, full_pad_size)

        return tensor

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __hash__(self) -> int:
        return hash(self.dim)

    def __repr__(self) -> str:
        return f"Shard(dim={self.dim})"

    def __str__(self) -> str:
        return f"S({self.dim})"


@dataclass(frozen=True)
class Replicate(Placement):
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Replicate):
            return False
        return True

    def __hash__(self) -> int:
        return -1

    def __repr__(self) -> str:
        return "Replicate()"

    def __str__(self) -> str:
        return "R"

    def _replicate_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        """
        Replicate (broadcast) a tensor on a mesh dimension use the first coordinate
        on the mesh dimension as source.
        """
        my_coordinate = mesh.get_coordinate()
        if my_coordinate is None:
            # If rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        tensor = tensor.contiguous()
        mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)

        return tensor


@dataclass(frozen=True)
class _Partial(Placement):
    reduce_op: ReduceOp.RedOpType = ReduceOp.SUM

    def _to_replicate(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
    ) -> torch.Tensor:
        return all_reduce(tensor, op=self.reduce_op, group=mesh.get_dim_groups(mesh_dim))

    def _to_shard(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # By defualt call `reduce_shard_tensor` of the `shard_spec`
        shard_spec = cast(Shard, shard_spec)
        return shard_spec._reduce_shard_tensor(tensor, mesh, self.reduce_op, mesh_dim)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __hash__(self) -> int:
        return 1 + hash(self.reduce_op)

    def __repr__(self) -> str:
        return f"_Partial(reduce_op={self.reduce_op})"

    def __str__(self) -> str:
        return "P"


class TensorMeta(NamedTuple):
    shape: torch.Size
    stride: Tuple[int, ...]
    dtype: torch.dtype


@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: Tuple[Placement, ...]

    # Be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None

    def __post_int__(self):
        if not isinstance(self.placements, tuple):
            self.placements = tuple(self.placements)
        self._hash: Optional[int] = None

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # Make sure to recompute the hash in case any of the hashed attributes change
        if hasattr(self, "_hash") and name in ("mesh", "placements", "tensor_meta"):
            self._hash = None

    def _hash_impl(self) -> int:
        # Hashing and equality check for `DTensorSpec` are used to cache the sharding propagation results
        # We only need to consider the mesh, placements, shape, dtype and stride
        if self.tensor_meta is not None:
            return hash(
                (
                    self.mesh,
                    self.placements,
                    self.tensor_meta.shape,
                    self.tensor_meta.stride,
                    self.tensor_meta.dtype,
                )
            )

        return hash((self.mesh, self.placements))

    def __hash__(self) -> int:
        # Lazily cache the spec to avoid recomputing the hash upon each use
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash

    def __eq__(self, value: object) -> bool:
        if (not isinstance(value, DTensorSpec)) and (self.mesh == value.mesh) and (self.placements == value.placements):
            return False
        if (self.tensor_meta is None) or (value.tensor_meta is None):
            return self.tensor_meta == value.tensor_meta

        return (
            (self.tensor_meta.shape == value.tensor_meta.shape)
            and (self.tensor_meta.stride == value.tensor_meta.stride)
            and (self.tensor_meta.dtype == value.tensor_meta.dtype)
        )

    def __str__(self) -> str:
        if len(self.placements) == 1:
            placement_str = str(self.placements[0])
        else:
            placement_str = str(self.placements)

        if self.tensor_meta is not None:
            tensor_shape = str(tuple(self.tensor_meta.shape))
        else:
            tensor_shape = "unknown shape"

        return f"Spec({placement_str} on {tensor_shape})"

    @property
    def shape(self) -> torch.Size:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def ndim(self) -> int:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        num_shards = 1
        for i, placement in enumerate(self.placements):
            if placement.is_shard():
                num_shards *= self.mesh.size(i)

        return num_shards

    @property
    def dim_map(self) -> List[int]:
        """
        `dim_map` is a property we derive from `placements` of the distributed tensor.
        It simply return a list of ints where `dim_map[i]` denotes the sharding mapping to the mesh dimension,
        and `len(dim_map) == dist_tensor.ndim`.

        - `dim_map[i] == -1` means tensor dim i replicate on mesh.
        - `dim_map[i] == j` means tensor dim i shard on mesh dim j.

        Note that if placements contains `_Partial`, we have to explicitly deal with it, so that when we create
        a DTensorSpec with `dim_map`, we could properly record the pending sum.
        """
        r = [-1] * self.ndim

        for i, placemet in enumerate(self.placements):
            if placemet.is_shard():
                shard_dim = cast(Shard, placemet).dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet"
                    )
                r[shard_dim] = i

        return r

    @property
    def sums(self) -> List[int]:
        """
        Returns a list of ints where `sums[i]` denotes the pending sum (partial) on mesh dim i.
        """
        return [idx for idx, placement in enumerate(self.placements) if placement.is_partial()]

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: List[int],
        sums: List[int],
        tensor_meta: Optional[TensorMeta] = None,
    ) -> "DTensorSpec":
        """
        Construct a `DTensorSpec` from `dim_map` list and pending sum.

        Args:
            mesh: `DeviceMesh`
                device mesh to be used in the DTensorSpec
            dim_map: List[int]
                a list of integer that represents sharding on each tensor dimension.
                See `dim_map`
            sums: List[int]
                a list of integer that represents the distributed tensor have pending sum
                on which device mesh dimension
            tensor_meta: TensorMeta
                DTensor meta-data
        """
        # By default replicate on device mesh dimensions
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # Find all mesh dimensions that need pending reductions
        for s in sums:
            placements[s] = _Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        "DeviceMesh dimension can't be mapped to"
                        f" two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(f"DeviceMesh dimension {m} can't be both shard and partial")
                placements[m] = Shard(i)

        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self):
        """
        Returns `True` if the current `DTensorSpec` replicates on all mesh dimensions.
        """
        return all(placement.is_replicate() for placement in self.placements)
