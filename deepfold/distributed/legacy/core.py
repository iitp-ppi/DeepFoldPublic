# Copyright 2023 Deepfold Team

import importlib
from typing import Any, Optional

import torch.distributed as dist

_Group = Optional[Any]

# Data parallel group that the current rank belongs to.
DATA_PARALLEL_GROUP: _Group = None

# Intra-layer model parallel group that the current rank belongs to.
TENSOR_MODEL_PARALLEL_GROUP: _Group = None


def _ensure_divisibility(n: int, d: int) -> None:
    """
    Ensure the numerator is divisible by the denominator.
    """
    assert n % d == 0, f"{n} is not divisible by {d}"


def init_distributed(
    world_size: int = -1,
    rank: int = -1,
    tensor_model_parallel_size: int = 1,
    dist_engine: str = "torch",
    dist_backend: str = "nccl",
) -> None:
    """
    Initialize the distributed environment.

    The following environment variables must be defined.
    - MASTER_PORT
    - MASTER_ADDR
    - WORLD_SIZE
    - RANK
    """
    dist_backend = dist_backend.lower()
    dist_engine = dist_engine.lower()

    if dist_engine == "deepspeed":
        deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
        if deepspeed_is_installed:
            global deepspeed
            import deepspeed

            deepspeed.init_distributed(dist_backend=dist_backend)
        else:
            raise ValueError("DeepSpeed is not installed")
    elif dist_engine == "torch":
        dist.init_process_group(
            backend=dist_backend,
            world_size=world_size,
            rank=rank,
        )
    else:
        raise ValueError(f"{dist_engine} is not supported")

    if tensor_model_parallel_size == -1:
        tensor_model_parallel_size = dist.get_world_size()

    # Check dist configs
    _ensure_divisibility(world_size, tensor_model_parallel_size)
    data_parallel_size = world_size // tensor_model_parallel_size

    # If you want to modify the way rank allocated,
    # you must also modify `get_tensor_model_parallel_rank()`.

    # Build the data parallel groups
    global DATA_PARALLEL_GROUP
    assert DATA_PARALLEL_GROUP is None
    for i in range(tensor_model_parallel_size):
        ranks = list(range(i, world_size, tensor_model_parallel_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            DATA_PARALLEL_GROUP = group

    # Build the model parallel groups
    global TENSOR_MODEL_PARALLEL_GROUP
    assert TENSOR_MODEL_PARALLEL_GROUP is None
    for i in range(data_parallel_size):
        ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            TENSOR_MODEL_PARALLEL_GROUP = group

    # if dist.get_rank() == 0: # TODO: Rank zero only logger needed
    #     print(f"> Initialize tensor parallel with size {tensor_model_parallel_size}")
    #     print(f"> Initialize data parallel with size {data_parallel_size}")


def get_tensor_model_parallel_group():
    """
    Get the tensor model parallel group the caller ank belongs to.
    """
    assert TENSOR_MODEL_PARALLEL_GROUP is not None
    return TENSOR_MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """
    Return world size for the tensor model parallel group.
    """
    return dist.get_world_size(group=get_tensor_model_parallel_group())


def get_tensor_model_parallel_rank():
    """
    Return my rank for the tensor model parallel group.
    """
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_data_parallel_group():
    """
    Get the tensor model parallel group the caller ank belongs to.
    """
    assert DATA_PARALLEL_GROUP is not None
    return DATA_PARALLEL_GROUP


def get_data_parallel_world_size():
    """
    Return world size for the tensor model parallel group.
    """
    return dist.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """
    Return my rank for the tensor model parallel group.
    """
    return dist.get_rank(group=get_data_parallel_group())


def get_tensor_model_parallel_master_rank():
    """
    Calculate the global rank corresponding to the first local rank in the
    tensor model parallel group.
    """
    global_rank = dist.get_rank()
    local_world_rank = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_rank) * local_world_rank
