# Copyright 2024 DeepFold Team


import contextlib
import random
from typing import Optional

import numpy as np

NUMPY_SEED_MODULUS = 0xFFFF_FFFF + 1
TORCH_SEED_MODULUS = 0xFFFF_FFFF_FFFF_FFFF + 1


def str_hash(string: str):
    hash = 0
    for ch in string:
        hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return hash


@contextlib.contextmanager
def numpy_seed(seed: Optional[int] = None, *additional_seeds, key: str = ""):
    """
    Context manager which seeds the Numpy PRNG with the seed and restores the state.
    """

    if seed is None:
        yield
        return

    if len(additional_seeds) > 0:
        additional_seeds = [int(i) for i in additional_seeds]
        seed = hash((seed, *additional_seeds)) % 100000000

    if key is not None:
        seed = hash((seed, str_hash(key))) % 100000000

    state = np.random.get_state()
    np.random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(state)


def get_seed_from_string(s: str) -> int:
    """Hashes input string and returns uint64-like integer seed value."""
    rng = random.Random(s)
    seed = rng.getrandbits(64)
    return seed


def get_seed_randomly() -> int:
    """Returns truly pseduorandom uint64-like integer seed value."""
    rng = random.Random(None)
    seed = rng.getrandbits(64)
    return seed
