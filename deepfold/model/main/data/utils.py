# Copyright 2024 DeepFold Team


import copy as copy_lib
import functools
import gzip
import pickle
from typing import Any, Callable

import numpy as np

from deepfold.common import residue_constants as rc


def lru_cache(maxsize: int = 16, typed: bool = False, copy: bool = False):
    if copy:

        def decorator(f: Callable):
            cached_func = functools.lru_cache(maxsize=maxsize, typed=typed)(f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return copy_lib.copy(cached_func(*args, **kwargs))

            return wrapper

    else:
        decorator = functools.lru_cache(maxsize=maxsize, typed=typed)

    return decorator


@lru_cache(maxsize=8, copy=True)
def load_pickle(path: str) -> Any:
    """Load a pickle file or gzipped pickle file."""

    def load(path: str):
        assert path.endswith(".pkl") or path.endswith(".pkl.gz"), f"Bad suffix in '{path}' as pickle file"
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, "rb") as fp:
            return pickle.load(fp)

    ret = load(path)

    return ret
