# Copyright 2024 DeepFold Team


"""Utility functions."""


import operator
from functools import reduce

import torch


def ensure_divisibility(numerator: int, denominator: int):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class GlobalMemoryBuffer:
    """
    Global buffer to avoid dynamic memory allocations.

    Note:
        Caller should ensure that buffers of the same name are not used concurrently.
    """

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def assert_viewless_tensor(tensor):
    """Assert that a tensor is not a view."""

    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, "Ensure tensor._base is None before setting tensor.data"

    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    """Safely set tensor's `.data` field."""
    assert_viewless_tensor(tensor)
    tensor.data = new_data_tensor
