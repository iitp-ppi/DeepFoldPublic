import hashlib


def hash_string_into_number(s: str) -> int:
    """Hashes string into uint64-like integer number."""
    b = s.encode("utf-8")
    d = hashlib.sha256(b).digest()
    i = int.from_bytes(d[:8], byteorder="little", signed=False)
    return i
