from typing import Iterator, Tuple


def slice_generator(start: int, end: int, size: int) -> Iterator[Tuple[int, int]]:
    """Returns slice indices iterator from start to end."""
    for i in range(start, end, size):
        left = i
        right = min(i + size, end)
        yield left, right
