import logging
import os
from pathlib import Path


class TqdmHandler(logging.StreamHandler):
    def __init__(self) -> None:
        logging.StreamHandler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm import tqdm

        msg = self.format(record)
        tqdm.write(msg)


def setup_logging(filename: os.PathLike, mode: str = "w") -> None:
    assert mode in ("w", "a")  # Not read mode

    # Make parent directory
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)

    # Setup root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            handler.close()
            root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[TqdmHandler(), logging.FileHandler(filename=filename, mode=mode)],
        force=True,
    )


def safe_filename(name: str) -> str:
    return "".join([c if c.isalnum() or c in ("_", ".", "-") else "_" for c in name])
