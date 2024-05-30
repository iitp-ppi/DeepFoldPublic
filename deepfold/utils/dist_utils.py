import torch


def pad_tensor(tensor: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    if pad_size == 0:
        return tensor
    if dim < 0:
        pad = [0, 0] * -dim
    else:
        pad = [0, 0] * (tensor.ndim - dim)
    pad[-1] = pad_size
    return torch.nn.functional.pad(tensor, pad, mode="constant", value=0.0)


def get_pad_size(tensor: torch.Tensor, dim: int, num_chunks: int) -> int:
    seq_len = tensor.size(dim)
    chunk_size = (seq_len + num_chunks - 1) // num_chunks
    return num_chunks * chunk_size - seq_len
