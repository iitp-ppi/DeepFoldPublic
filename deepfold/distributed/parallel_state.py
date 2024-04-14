# Copyright 2024 DeepFold Team


"""Model and data parallel groups."""


from typing import List, Optional

import torch
import torch.distributed

from deepfold.distributed.utils import GlobalMemoryBuffer

# Model parallel group that the current rank belongs to
_MODEL_PARALLEL_GROUP = None


# Data parallel group that the current rank belongs to
_DATA_PARALLEL_GROUP = None


# A list of global ranks for each data parallel group
_DATA_PARALLEL_GLOBAL_RANKS: List[int] = None


# Model parallel group and data parallel group combined
_MODEL_AND_DATA_PARALLEL_GROUP = None


# Memory buffers to avoid dynamic memory allocation
_GLOBAL_MEMORY_BUFFER = None

# Whether model parallel has been initialized or not
_MODEL_PARALLEL_INITIALIZED = False

# Whether model parallel is enabled or not
_MODEL_PARALLEL_ENABLED = False


def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Arguments:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get("cga_cluster_size", 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get("max_ctas", 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get("min_ctas", 1)
        return nccl_options
    else:
        return None


def initialize_model_parallel(
    model_parallel_size: int = 1,
    rank: Optional[int] = None,
) -> None:
    """
    Initialize model-data parallel groups.

    Arguments:
        model_parallel_size (int, default = 1):
            The number of GPUs to split the model across.
    """

    # Get world size and rank
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % model_parallel_size != 0:
        raise RuntimeError(f"World size ({world_size}) is not divisible by model parallel size ({model_parallel_size})")

    # data_parallel_size: int = world_size // model_parallel_size
    num_model_parallel_groups: int = world_size // model_parallel_size
    rank = rank if rank is not None else torch.distributed.get_rank()

    # Build the model-parallel groups
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, "Model parallel group is already initialized"
    for i in range(num_model_parallel_groups):
        start_rank = i * model_parallel_size
        end_rank = (i + 1) * model_parallel_size
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the data-parallel groups
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, "Data parallel group is already initialized"
    for i in range(model_parallel_size):
        ranks = range(i, world_size, model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GLOBAL_RANKS = list(ranks)

    # Build the model and data parallel groups
    global _MODEL_AND_DATA_PARALLEL_GROUP
    assert _MODEL_AND_DATA_PARALLEL_GROUP is None, "Model and data parallel group is already initialized"
    ranks = range(world_size)
    group = torch.distributed.new_group(ranks)
    if rank in ranks:
        _MODEL_AND_DATA_PARALLEL_GROUP = group

    _set_global_memory_buffer()

    _MODEL_PARALLEL_INITIALIZED = True
    _MODEL_PARALLEL_ENABLED = True


def model_parallel_is_initialized() -> bool:
    """Check if model and data parallel groups are initialized."""
    if _MODEL_PARALLEL_GROUP is None or _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Returns the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "Model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Returns the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "Data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_model_and_data_parallel_group():
    """Returns the tensor and data parallel group the caller rank belongs to."""
    assert _MODEL_AND_DATA_PARALLEL_GROUP is not None, "Model and data parallel group is not initialized"
    return _MODEL_AND_DATA_PARALLEL_GROUP


def get_model_parallel_rank() -> int:
    """Returns my rank for the model parallel group."""
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_world_size() -> int:
    """Returns world size for the tensor model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def get_model_parallel_src_rank() -> int:
    """Returns the global rank corresponding to the first local rank in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank() -> int:
    """Returns the global rank corresponding to the first local rank."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_data_parallel_world_size() -> int:
    """Returns world size for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_data_parallel_group())
    else:
        return 0


def get_data_parallel_rank() -> int:
    """Returns my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize the global buffer."""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, "Global memory buffer is already initialized"
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Returns the global memory buffer object."""
    assert _GLOBAL_MEMORY_BUFFER is not None, "Global memory buffer is not initialized"
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None."""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to None."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _MODEL_AND_DATA_PARALLEL_GROUP
    _MODEL_AND_DATA_PARALLEL_GROUP = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def is_initialized() -> bool:
    return _MODEL_PARALLEL_INITIALIZED


def is_enabled() -> bool:
    return _MODEL_PARALLEL_ENABLED


def _enable() -> None:
    global _MODEL_PARALLEL_ENABLED
    _MODEL_PARALLEL_ENABLED = True


def _disable() -> None:
    global _MODEL_PARALLEL_ENABLED
    _MODEL_PARALLEL_ENABLED = False


class Enable(torch.autograd.Function):

    @staticmethod
    def forward(ctx: "Enable") -> None:
        return _enable()

    @staticmethod
    def backward(ctx: "Enable") -> None:
        return _enable()


class Disable(torch.autograd.Function):
    @staticmethod
    def forward(ctx: "Disable") -> None:
        return _disable()

    @staticmethod
    def backward(ctx: "Disable") -> None:
        return _enable()


def enable() -> None:
    if is_initialized():
        if torch.is_grad_enabled():
            Enable.apply()
        else:
            _enable()


def disable() -> None:
    if is_initialized():
        if torch.is_grad_enabled():
            Disable.apply()
        else:
            _disable()


# Shortcuts
rank = get_model_parallel_rank
size = get_model_parallel_world_size
