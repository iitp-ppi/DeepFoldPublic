import itertools
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Generator, Iterator, List, Sequence, Tuple, TypeVar, cast

import torch
import torch.distributed as dist

from deepfold.distributed.tensor import Mesh, Replicate, Shard, distribute_tensor, redistribute
from deepfold.distributed.tensor.api import DTensor
from deepfold.distributed.tensor.deps import pytree
from deepfold.distributed.tensor.placements import Placement

from .common_distributed import TEST_SKIPS, MultiProcessTestCase

DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
NUM_DEVICES = 4

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

T = TypeVar("T")


class MLPModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(10, 16, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(16, 12, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


@dataclass
class RedistributeProfile:
    num_calls: int


@contextmanager
def redistribute_profiler() -> Generator[RedistributeProfile, None, None]:
    orig_redistribute_dtensor = redistribute.redistribute_dtensor
    profile: RedistributeProfile = RedistributeProfile(num_calls=0)

    def patched_redistribute_dtensor(
        input: DTensor,
        mesh: Mesh,
        placements: Sequence[Placement],
    ) -> DTensor:
        result = orig_redistribute_dtensor(input, mesh, placements)
        profile.num_calls += 1
        return result

    try:
        redistribute.redistribute_dtensor = patched_redistribute_dtensor
        yield profile
    finally:
        redistribute.redistribute_dtensor = orig_redistribute_dtensor


class DTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    def build_mesh(self) -> Mesh:
        return Mesh(DEVICE_TYPE, list(range(NUM_DEVICES)))

    def init_pg(self, backend: str = "nccl") -> None:
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS)

        if backend not in ["nccl", "gloo", "mpi"]:
            raise RuntimeError(f"Backend {backend} not supported")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _test_op(self, mesh: Mesh, op_call, *args, **kwargs) -> None:
        with redistribute_profiler() as profile:
            out = op_call(*args, **kwargs)
            dtc = DTensorConverter(mesh, args, kwargs)
            for d_args, d_kwargs in dtc:
                self.assertEqual(dtc.successful(), True)
                d_out = op_call(*d_args, **d_kwargs)
                self.assertEqual(
                    d_out.redistribute(mesh, [Replicate() for _ in range(mesh.ndim)]).to_local(),
                    out,
                )


# This is a class for converting args/kwargs of an op into distributed args/kwargs
class DTensorConverter:
    def __init__(
        self,
        mesh: Mesh,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        flatten_args, flatten_args_spec = pytree.tree_flatten(args)
        flatten_kwargs, flatten_kwargs_spec = pytree.tree_flatten(kwargs)

        self.flatten_args: List[object] = flatten_args
        self.flatten_args_spec: pytree.TreeSpec = flatten_args_spec
        self.flatten_kwargs: List[object] = flatten_kwargs
        self.flatten_kwargs_spec: pytree.TreeSpec = flatten_kwargs_spec

        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        self.sharding_combs: Iterator[Sequence[Placement]] = iter(itertools.product(*choices_for_args))

    def successful(self) -> bool:
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        # TODO: dist tensor need to support quantized and sparse
        # tensors, quantized tensor might be relatively easy, but
        # sparse tensor have special layouts that we need to possibly
        # deal with, until we are clear about them, we don't officially
        # support them.
        return not any(
            [
                t.is_sparse_csr,
                t.is_sparse,
                t.is_mkldnn,
                t.is_quantized,
                t.is_nested,
                torch._is_functional_tensor(t),
                t.is_neg(),
                t.is_conj(),
                t.device.type in ("lazy", "meta"),
                # We need a way to test if a tensor is batched but there
                # is no official APi to do it
                # torch._C._is_batched(t),
            ]
        )

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        mesh_size = self.mesh.size()
        sharding_choices: List[Placement] = [Replicate()]
        # c10d collective does not support bool tensor
        # for bool tensor we treat it as replicated
        if arg.dtype != torch.bool:
            # only generating choices with: replicate, or sharding
            # evenly on a dimension that could be sharded
            sharding_choices = sharding_choices + [
                Shard(i) for i, s in enumerate(arg.shape) if s > 1 and s % mesh_size == 0
            ]
        # TODO: add multi mesh choices
        # all_choices = itertools.product(
        #     *(self.mesh.ndim * [sharding_choices])
        # )
        return sharding_choices

    def __iter__(self) -> "DTensorConverter":
        return self

    def __next__(self) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            new_args: List[object] = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_args.append(arg)

            new_kwargs: List[object] = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_kwargs.append(arg)

            return (
                pytree.tree_unflatten(new_args, self.flatten_args_spec),
                pytree.tree_unflatten(new_kwargs, self.flatten_kwargs_spec),
            )
        except StopIteration as e:
            raise StopIteration from e

    def to_dist_tensor(self, t: torch.Tensor, mesh: Mesh, placements: List[Placement]) -> torch.Tensor:
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            if self.is_supported_tensor(t):
                self.hit += 1
                # We cannot use distribute_tensor for bool tensors as c10d
                # collectives does not support the dtype, we assume op with
                # bool tensor args the same tensor so we don't need to broadcast
                # TODO: add bool tensor dtype support in c10d collective
                if t.dtype == torch.bool:
                    r = DTensor(
                        t,
                        mesh,
                        placements,
                        size=t.size(),
                        dtype=torch.bool,
                        requires_grad=t.requires_grad,
                        stride=t.stride(),
                    )
                else:
                    r = distribute_tensor(t, mesh, placements)
                if type(t) is torch.nn.Parameter:
                    r = torch.nn.Parameter(r, requires_grad=r.requires_grad)  # type: ignore[assignment]
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to dist tensor can cause
            # unpredictable problems, we explicitly disable this conversion
            # for now (i.e. we don't support DTensor holding tensor subclass
            # until there's a strong reason later).
            self.miss += 1
            return t
        else:
            raise RuntimeError(f"Trying to convert to DTensor, but got {type(t)}")


TestFunc = Callable[[Any], Any]


# wrapper to initialize comms (processgroup)
def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
        # if backend not specified, and cuda available, then use nccl, else gloo
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"

        pg_backend = "nccl" if self.device_type == "cuda" else "gloo"
        if pg_backend == "nccl" and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.init_pg(backend=pg_backend)
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

    return wrapper
