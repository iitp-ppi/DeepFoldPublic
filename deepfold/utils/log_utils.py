import json
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd


class TqdmHandler(logging.StreamHandler):
    def __init__(self) -> None:
        logging.StreamHandler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        from tqdm.auto import tqdm

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


def save_logs(logs: List[dict], outpath: Path, append: bool) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for log in logs:
        line = json.dumps(log)
        lines.append(line)
    outstr = "\n".join(lines) + "\n"
    mode = "a" if append else "w"
    with open(outpath, mode) as f:
        f.write(outstr)


def read_logs(
    filepath: Path,
    drop_overridden_iterations: bool = True,
) -> pd.DataFrame:
    with open(filepath) as f:
        logs = f.read().strip().split("\n")
    logs = [json.loads(log) for log in logs]
    logs_df = pd.DataFrame(logs)
    if drop_overridden_iterations:
        logs_df = logs_df.drop_duplicates("iteration", keep="last")
        logs_df = logs_df.reset_index(drop=True).copy()
    return logs_df
