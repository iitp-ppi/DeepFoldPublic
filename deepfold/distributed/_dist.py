from typing import Dict, List, Optional

import torch
import torch.distributed

# whether torch distributed has been initialized or not
_DIST_INITIALIZED = False

# distributed rank, from 0 to world_size-1
_DIST_RANK = None

# distributed world size
_DIST_WORLD_SIZE = None

# num ranks (device count) per node
_DIST_LOCAL_WORLD_SIZE = None

# rank inside node (device id)
_DIST_LOCAL_RANK = None

# number of nodes
_DIST_NUM_NODES = None

# list of train ranks
_DIST_TRAIN_RANKS = []

# training process group
_DIST_TRAIN_PROCESS_GROUP = None

# validation process group
_DIST_VAL_PROCESS_GROUP = None

# main training rank
_DIST_MASTER_RANK = None


def initialize() -> None:
    global _DIST_INITIALIZED
    global _DIST_RANK
    global _DIST_WORLD_SIZE
    global _DIST_LOCAL_WORLD_SIZE
    global _DIST_LOCAL_RANK
    global _DIST_NUM_NODES
    global _DIST_TRAIN_RANKS
    global _DIST_MASTER_RANK

    assert not _DIST_INITIALIZED
    assert torch.distributed.is_available()
    assert not torch.distributed.is_initialized()

    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    assert torch.distributed.is_initialized()

    rank = int(torch.distributed.get_rank())
    world_size = int(torch.distributed.get_world_size())

    local_world_size = int(torch.cuda.device_count())
    local_rank = rank % local_world_size

    num_nodes = int(max(1, world_size // local_world_size))
    if world_size >= local_world_size:
        assert world_size % local_world_size == 0
    else:
        assert num_nodes == 1

    _DIST_RANK = rank
    _DIST_WORLD_SIZE = world_size
    _DIST_LOCAL_WORLD_SIZE = local_world_size
    _DIST_LOCAL_RANK = local_rank
    _DIST_NUM_NODES = num_nodes
    _DIST_TRAIN_RANKS = list(range(world_size))
    _DIST_MASTER_RANK = 0
    _DIST_INITIALIZED = True


def is_initialized() -> bool:
    return _DIST_INITIALIZED


def rank() -> Optional[int]:
    return _DIST_RANK


def world_size() -> Optional[int]:
    return _DIST_WORLD_SIZE


def local_world_size() -> Optional[int]:
    return _DIST_LOCAL_WORLD_SIZE


def local_rank() -> Optional[int]:
    return _DIST_LOCAL_RANK


def num_nodes() -> Optional[int]:
    return _DIST_NUM_NODES


def train_ranks() -> List[int]:
    return _DIST_TRAIN_RANKS


def num_train_ranks() -> int:
    return len(_DIST_TRAIN_RANKS)


def train_process_group() -> Optional[torch.distributed.ProcessGroup]:
    return _DIST_TRAIN_PROCESS_GROUP


def val_process_group() -> Optional[torch.distributed.ProcessGroup]:
    return _DIST_VAL_PROCESS_GROUP


def master_rank() -> Optional[int]:
    return _DIST_MASTER_RANK


def is_master_process() -> bool:
    return not _DIST_INITIALIZED or bool(_DIST_RANK == _DIST_MASTER_RANK)


is_main_process = is_master_process


def is_train_rank() -> bool:
    return not _DIST_INITIALIZED or _DIST_RANK in _DIST_TRAIN_RANKS


def reduce_losses_avg(
    losses: Dict[str, torch.Tensor],
    device: torch.device,
    synchronize: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """Reduces to average losses from all train processes.

    All input `losses` must have the same order of keys.

    Returns averaged values in the main process, otherwise `None`.

    """
    if not _DIST_INITIALIZED:
        return losses
    reduce_tensor = torch.stack(list(losses.values()))
    assert reduce_tensor.ndim == 1
    reduce_tensor = reduce_tensor.to(device=device, dtype=torch.float32)
    torch.distributed.reduce(
        tensor=reduce_tensor,
        dst=_DIST_MASTER_RANK,
        op=torch.distributed.ReduceOp.AVG,
        group=_DIST_TRAIN_PROCESS_GROUP,
    )
    if synchronize:
        torch.distributed.barrier(group=_DIST_TRAIN_PROCESS_GROUP)
    if _DIST_RANK != _DIST_MASTER_RANK:
        return None
    losses_avg = {}
    i = 0
    for key in losses.keys():
        losses_avg[key] = reduce_tensor[i]
        i += 1
    return losses_avg


def gather_val_metrics(
    val_metrics_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
    synchronize: bool,
) -> Optional[List[dict]]:
    """Gathers validation metrics to main val process.

    All dictionaries inside `val_metrics_list` must have the same order of keys.

    Returns gathered values in the main val process, otherwise `None`.

    """
    if not _DIST_INITIALIZED:
        return val_metrics_list
    if len(val_metrics_list) == 0:
        val_metrics_list.append({})
    keys = list(val_metrics_list[0].keys())
    vtypes = {key: type(value) for key, value in val_metrics_list[0].items()}
    if _DIST_RANK == _DIST_MASTER_RANK:
        assert len(vtypes) > 0
    gather_nrows = len(val_metrics_list)
    gather_ncols = len(keys) + 1
    gather_shape = [gather_nrows, gather_ncols]
    all_reduce_tensor = torch.tensor(gather_shape, device=device)  # LongTensor
    torch.distributed.all_reduce(
        tensor=all_reduce_tensor,
        op=torch.distributed.ReduceOp.MAX,
        group=_DIST_VAL_PROCESS_GROUP,
    )
    gather_shape[0] = all_reduce_tensor[0].item()
    gather_shape[1] = all_reduce_tensor[1].item()
    if 0 in gather_shape:
        raise RuntimeError(f"unexpected gather_shape={gather_shape}")
    gather_tensor = torch.zeros(size=gather_shape, dtype=torch.float64)
    for i, val_metrics in enumerate(val_metrics_list):
        if val_metrics:
            gather_tensor[i][0] = 1
        for j, key in enumerate(keys, 1):
            gather_tensor[i][j] = val_metrics[key]
    gather_tensor = gather_tensor.to(device=device)
    assert _DIST_MASTER_RANK is not None
    if _DIST_RANK == _DIST_MASTER_RANK:
        gather_list = [torch.zeros_like(gather_tensor) for _ in range(len(_DIST_TRAIN_RANKS))]
    else:
        gather_list = None
    torch.distributed.gather(
        tensor=gather_tensor,
        gather_list=gather_list,
        dst=_DIST_MASTER_RANK,
        group=_DIST_VAL_PROCESS_GROUP,
    )
    if synchronize:
        torch.distributed.barrier(group=_DIST_VAL_PROCESS_GROUP)
    if _DIST_RANK != _DIST_MASTER_RANK:
        return None
    assert gather_list is not None
    gather_list = [gather_tensor.cpu() for gather_tensor in gather_list]
    gather_val_metrics_list = []
    for gather_tensor in gather_list:
        for i in range(gather_shape[0]):
            if gather_tensor[i][0]:
                gather_val_metrics = {}
                for j, key in enumerate(keys, 1):
                    value = gather_tensor[i][j].item()
                    vtype = vtypes[key]
                    value = vtype(value)
                    gather_val_metrics[key] = value
                gather_val_metrics_list.append(gather_val_metrics)
    return gather_val_metrics_list
