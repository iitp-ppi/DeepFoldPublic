import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from deepfold.core import parallel_state as ps
from deepfold.core.model_parallel.random import checkpoint as dist_checkpoint


def get_checkpoint_fn():
    if ps.model_parallel_is_initialized():
        checkpoint = dist_checkpoint
    else:
        checkpoint = torch_checkpoint

    return checkpoint
