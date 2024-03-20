import inspect
import os
import sys
from functools import wraps

import torch
from torch.distributed import get_rank

_CNT = -1


def dump_args(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    @wraps(func)
    def identity(*args, **kwargs):
        return func(*args, **kwargs)

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _CNT
        if get_rank() == 0:
            _CNT += 1
            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            ts = {}
            for k, v in func_args.items():
                if isinstance(v, torch.Tensor):
                    ts[k] = tuple(v.shape)
            func_args_str = ", ".join(map("{0[0]}: {0[1]!r}".format, ts.items()))
            s = "│" * _CNT
            print(
                s,
                "┌",
                f"{func.__module__}.{func.__qualname__} <- ( {func_args_str} )",
                file=sys.stderr,
                sep="",
            )
        ret = func(*args, **kwargs)
        if get_rank() == 0:
            ts = []
            if isinstance(ret, tuple):
                for v in ret:
                    if isinstance(v, torch.Tensor):
                        ts.append(tuple(v.shape))
            elif isinstance(ret, torch.Tensor):
                ts.append((str(type(ret)), tuple(ret.shape)))
            func_ret_str = ", ".join(map("{0[0]!r}".format, ts))
            s = "│" * _CNT
            print(
                s,
                "└",
                f"{func.__module__}.{func.__qualname__} -> ( {func_ret_str} )",
                file=sys.stderr,
                sep="",
            )
            _CNT -= 1
        return ret

    if "DEBUG" in os.environ and os.environ["DEBUG"] != "0":
        return wrapper
    else:
        return identity
