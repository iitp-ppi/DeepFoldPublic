# DeepFold Team


import dataclasses
from functools import reduce
from operator import mul
from typing import List, Sequence


def get_field_names(cls) -> List[str]:
    fields = dataclasses.fields(cls)
    field_names = [f.name for f in fields]

    return field_names


def prod(x: Sequence[int]) -> int:
    return reduce(mul, x, 1)
