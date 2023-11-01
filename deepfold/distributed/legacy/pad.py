# Copyright 2023 Deepfold Team

from deepfold.distributed.legacy.comm import get_tensor_model_parallel_world_size


def pad_size(axis_size: int, world_size: int | None = None):
    if world_size is None:
        world_size = get_tensor_model_parallel_world_size()

    return (axis_size // world_size + 1) * world_size - axis_size
