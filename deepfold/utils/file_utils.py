import functools
import gzip
import os
import pickle
import warnings
from glob import glob
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np


def restore_wrapper(func):
    @functools.wraps(func.__wrapped__)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@restore_wrapper
@functools.lru_cache(maxsize=16)
def get_file_content_and_extension(file_path: os.PathLike) -> Tuple[str, str]:
    """
    Reads the content of a text file or compressed text file and returns its content
    along with the real extension (excluding compression extensions like .gz).

    Args:
        file_path (str): Path to the text file or compressed text file.

    Returns:
        tuple: A tuple containing the file content (str) and the real extension (str).
    """
    # Determine if the file is gzipped by checking the magic number
    with open(file_path, "rb") as f:
        magic_number = f.read(2)
    if magic_number == b"\x1f\x8b":
        # It's a gzipped file
        open_func = lambda x: gzip.open(x, "rt")
    else:
        # It's a regular text file
        open_func = lambda x: open(x, "rt")

    # Get the real extension by removing compression suffixes
    path = Path(file_path)
    compression_suffixes = [".gz"]
    suffixes = path.suffixes

    # Remove compression extensions from the end
    real_suffixes = []
    for s in reversed(suffixes):
        if s.lower() in compression_suffixes:
            continue
        else:
            real_suffixes.insert(0, s)
            break  # Stop after finding the real extension

    real_extension = "".join(real_suffixes) if real_suffixes else ""

    # Read the content of the file
    with open_func(file_path) as f:
        content = f.read()

    return str(content), str(real_extension)


def read_text(path: os.PathLike) -> str:
    return _read_text(str(path))


@functools.lru_cache(16, typed=False)
def _read_text(path: str) -> str:
    try:
        if os.path.splitext(path)[1] == ".gz":
            with gzip.open(path, "rb") as fp:
                return fp.read().decode()
        else:
            with gzip.open(f"{path}.gz", "rb") as fp:
                return fp.read().decode()
    except FileNotFoundError:
        with open(path, "r") as fp:
            return fp.read()


def load_pickle(path: os.PathLike) -> Any:
    try:
        ext = os.path.splitext(path)[1]
        if ext == ".gz":
            return _load_gz_pkl(path)
        elif ext == ".pkz":
            return _load_pkz(path)
        elif ext == ".npz":
            return np.load(path)
        else:
            return _load_pkl(path)
    except FileNotFoundError:
        raise


def _load_gz_pkl(path: os.PathLike) -> Any:
    with gzip.open(path, "rb") as fp:
        return pickle.load(fp)


def _load_pkz(path: os.PathLike) -> Any:
    return _load_gz_pkl(path)


def _load_pkl(path: os.PathLike) -> Any:
    with open(path, "rb") as fp:
        return pickle.load(fp)


def dump_pickle(obj: Any, path: os.PathLike, level: int = 5) -> None:
    f, ext = os.path.splitext(path)
    assert ext in (".pkl", ".pkz", ".gz", ".npz")

    if ext == ".npz":
        np.savez_compressed(path, **obj)
    else:
        if ext == ".pkl":
            path = f + ".pkz"
            warnings.warn(f"Write on '{path}'")
        with gzip.open(path, "wb", compresslevel=level) as fp:
            pickle.dump(obj, fp)


def find_paths(paths: Sequence[os.PathLike]) -> List[Path]:
    found_paths = set()
    for regex in paths:
        path_found = glob(str(regex))
        for p in path_found:
            found_paths.add(str(p))
    found_paths = list(found_paths)
    found_paths.sort()
    return [Path(p) for p in found_paths]
