import torch

_ENABLED = False


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def is_enabled_and_autograd_off() -> bool:
    return _ENABLED and not torch.is_grad_enabled()
