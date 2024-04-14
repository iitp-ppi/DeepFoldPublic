"""Utilities related to tensor operations."""

from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, TypeVar, Union, overload

import numpy as np
import torch


def add(m1: torch.Tensor, m2: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-num_dims] + (-1,))


def masked_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    eps: float = 1e-4,
    keepdim: bool = False,
) -> torch.Tensor:
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (eps + torch.sum(mask, dim=dim, keepdim=keepdim))


def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return torch.nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def pts_to_distogram(
    pts: torch.Tensor,
    min_bin: float = 2.2325,
    max_bin: float = 21.6875,
    num_bins: int = 64,
) -> torch.Tensor:
    boundaries = torch.linspace(min_bin, max_bin, steps=(num_bins - 1), device=pts.device)
    dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
    return torch.bucketize(dists, boundaries, right=False)


def dict_multimap(fn: Callable, dicts: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def batched_gather(
    data: torch.Tensor,
    inds: torch.Tensor,
    dim: int = 0,
    num_batch_dims: int = 0,
) -> torch.Tensor:
    ranges = []
    for i, s in enumerate(data.shape[:num_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - num_batch_dims)]
    remaining_dims[dim - num_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)

    return data[ranges]


T = TypeVar("T")


# With tree_map, a poor man's JAX tree_map
def dict_map(
    fn: Callable[[T], Any],
    dic: Dict[Any, Union[dict, list, tuple, T]],
    leaf_type: Type[T],
) -> Dict[Any, Union[dict, list, tuple, Any]]:
    new_dict: Dict[Any, Union[dict, list, tuple, Any]] = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


@overload
def tree_map(fn: Callable[[T], Any], tree: T, leaf_type: Type[T]) -> Any: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: dict, leaf_type: Type[T]) -> dict: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: list, leaf_type: Type[T]) -> list: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: tuple, leaf_type: Type[T]) -> tuple: ...


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple(tree_map(fn, x, leaf_type) for x in tree)
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError("Not supported")


array_tree_map = partial(tree_map, leaf_type=np.ndarray)
tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
