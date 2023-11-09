# Copyright 2023 Deepfold Team


from deepfold.distributed.legacy.comm import (
    broadcast_async_begin,
    broadcast_async_end,
    broadcast_sync,
    col_to_row,
    gather,
    gather_async_begin,
    gather_async_end,
    identity,
    reduce,
    row_to_col,
    scatter,
)
from deepfold.distributed.legacy.core import (
    DATA_PARALLEL_GROUP,
    TENSOR_MODEL_PARALLEL_GROUP,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_master_rank,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    init_distributed,
)
from deepfold.distributed.legacy.shard import get_pad_size

__all__ = [
    "broadcast_async_begin",
    "broadcast_async_end",
    "broadcast_sync",
    "col_to_row",
    "gather",
    "gather_async_begin",
    "gather_async_end",
    "identity",
    "reduce",
    "row_to_col",
    "scatter",
    "DATA_PARALLEL_GROUP",
    "TENSOR_MODEL_PARALLEL_GROUP",
    "get_data_parallel_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "get_tensor_model_parallel_group",
    "get_tensor_model_parallel_master_rank",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "init_distributed",
    "get_pad_size",
    "get_rank",
    "get_world_size",
    "is_master",
]


def get_rank():
    return get_tensor_model_parallel_rank()


def get_world_size():
    return get_tensor_model_parallel_world_size()


def is_master():
    return get_rank() == get_tensor_model_parallel_master_rank()
