_ENABLED = False


def enable() -> None:
    global _ENABLED
    _ENABLED = True


def disable():
    global _ENABLED
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED
