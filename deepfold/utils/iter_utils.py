from typing import Callable, Iterator, MutableMapping, Tuple


def slice_generator(start: int, end: int, size: int) -> Iterator[Tuple[int, int]]:
    """Returns slice indices iterator from start to end."""
    for i in range(start, end, size):
        left = i
        right = min(i + size, end)
        yield left, right


def map_dict_values(fn: Callable, d: dict) -> dict:
    """Maps dictionary values using given function."""
    return {k: fn(v) for k, v in d.items()}


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))
