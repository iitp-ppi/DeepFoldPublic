# Copyright 2023 Deepfold Team

from typing import List, Optional, Tuple

import torch


def pad_tensor(
    tensor: torch.Tensor,
    dim: int,
    pad_size: int,
) -> torch.Tensor:
    pad = [0, 0] * (tensor.ndim - dim)
    pad[-1] = pad_size
    return torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)


def get_pad_size(tensor: torch.Tensor, dim: int, num_chunks: int) -> int:
    chunk_size = (tensor.size(dim) + num_chunks - 1) // num_chunks
    return num_chunks * chunk_size - tensor.size(dim)
