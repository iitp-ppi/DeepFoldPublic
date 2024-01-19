# Copyright 2024 DeepFold Team


import contextlib
import numpy as np
from typing import Optional


def str_hash(string: str):
    hash = 0
    for ch in string:
        hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return hash


@contextlib.contextmanager
def numpy_seed(seed: Optional[int] = None, key: str = ""):
    """
    Context manager which seeds the Numpy PRNG with the seed and restores the state.
    """

    if seed is None:
        yield
        return

    if key is not None:
        seed = hash((seed, str_hash(key))) % 1e8

    state = np.random.get_state()
    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)
