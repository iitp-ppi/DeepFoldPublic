import math
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch

from deepfold.utils.iter_utils import slice_generator
from deepfold.utils.tensor_utils import tensor_tree_map


def get_device_mem(device: str) -> float:
    if device != "cpu" and torch.cuda.is_available():
        dev_id = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(f"cuda:{dev_id}")
        total_mem_in_gigabyte = prop.total_memory / 1024 / 1024 / 1024
        return total_mem_in_gigabyte
    else:
        return 40.0  # Fallback


def automatic_chunk_size(
    seq_len: int,
    device: str | torch.device,
    is_16bit: bool = False,
) -> Tuple[int, int | None]:
    total_mem_in_gigabyte = get_device_mem(device)
    # From Uni-Fold
    # TODO: Need to calibrate
    factor = math.sqrt(total_mem_in_gigabyte / 40.0 * (0.55 * is_16bit + 0.45)) * 0.95
    if seq_len < int(1024 * factor):
        chunk_size = 256
        block_size = None
    elif seq_len < int(2048 * factor):
        chunk_size = 128
        block_size = None
    elif seq_len < int(3072 * factor):
        chunk_size = 64
        block_size = None
    elif seq_len < int(4096 * factor):
        chunk_size = 32
        block_size = 512
    else:
        chunk_size = 4
        block_size = 256
    return chunk_size, block_size


def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    num_batch_dims: int,
    out: Any | None = None,
    add_into_out: bool = False,
) -> Any:
    if not (len(inputs > 0)):
        raise ValueError("Must provide at least one input")

    def _dict_get_shapes(inputs: Dict[str, Any]) -> List[List[int]]:
        shapes = []
        if type(inputs) is torch.Tensor:
            shapes.append(list(input.shape))
        elif type(inputs) is dict:
            for v in inputs.values():
                shapes.extend(_dict_get_shapes(v))
        elif isinstance(inputs, Iterable):
            for v in inputs:
                shapes.extend(_dict_get_shapes(v))
        else:
            raise ValueError("Not supported")

    inputs = {k: v for k, v in inputs.items() if v is not None}
    initial_dims = [shape[:num_batch_dims] for shape in _dict_get_shapes(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    flat_batch_dim = math.prod(orig_batch_dims)

    def _flat_inputs(t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *t.shape[num_batch_dims:])
        assert t.shape[0] == flat_batch_dim or t.shape[0] == 1
        return t

    flat_inputs = tensor_tree_map(_flat_inputs, inputs)

    if out is not None:
        out = tensor_tree_map(_flat_inputs, out)

    for start, end in slice_generator(0, flat_batch_dim, chunk_size):

        def select_chunk(t: torch.Tensor) -> torch.Tensor:
            if t.shape[0] == 1:
                return t[0:1]
            else:
                return t[start, end]

        chunks = tensor_tree_map(select_chunk, flat_inputs)
        output_chunk = layer(**chunks)

        if out is None:
            _allocate = lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:])
            out = tensor_tree_map(_allocate, output_chunk)

        out_type = type(output_chunk)
        if out_type is tuple:
            for x1, x2 in zip(out, output_chunk):
                if add_into_out:
                    x1[start:end] += x2
                else:
                    x1[start:end] = x2
        elif out_type is torch.Tensor:
            if add_into_out:
                out[start:end] += output_chunk
            else:
                out[start:end] = output_chunk
        else:
            raise ValueError("Not supported")

    reshape = lambda t: t.view(orig_batch_dims + t.shape[1:])
    out = tensor_tree_map(reshape, out)

    return out
