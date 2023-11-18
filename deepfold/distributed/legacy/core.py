# Copyright 2023 Deepfold Team

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

_Group = Optional[dist.ProcessGroup]

logger = logging.getLogger(__name__)

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
    tensor_model_parallel_size: Optional[int] = None,
    random_seed: int = 12345,
) -> None:
    """
    Initialize the distributed environment.

    The following environment variables must be defined.
    - MASTER_PORT
    - MASTER_ADDR
    - WORLD_SIZE
    - RANK
    - LOCAL_RANK
    """

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # master_addr = os.environ["MASTER_ADDR"]
    # master_port = int(os.environ["MASTER_PORT"])

    if tensor_model_parallel_size is None:
        tensor_model_parallel_size = 1

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    if tensor_model_parallel_size is None:
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

    if world_size > 1:
        devcie_id = local_rank
        logger.debug(f"[{rank}] Set CUDA device to {devcie_id}")
        torch.cuda.set_device(device=devcie_id)

    logger.debug(f"[{rank}] Set random seed to {random_seed}")
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.synchronize()


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
