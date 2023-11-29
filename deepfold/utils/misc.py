# DeepFold Team


import dataclasses
import math
from typing import List, Sequence


def get_field_names(cls) -> List[str]:
    fields = dataclasses.fields(cls)
    field_names = [f.name for f in fields]

    return field_names


def prod(x: Sequence[int]) -> int:
    return math.prod(x)
