import json
import logging
import os
import re
import time
from typing import Dict

import torch

logger = logging.getLogger(__name__)


def get_basename(path: os.PathLike):
    return os.path.splitext(os.path.basename(os.path.normpath(path)))[0]


def make_output_directory(output_dir: os.PathLike, model_name: str):
    pred_dir = os.path.join(output_dir, model_name)
    os.makedirs(pred_dir, exist_ok=True)

    return pred_dir


def parse_fasta(fasta_str: str):
    fasta_str = re.sub(">$", "", fasta_str, flags=re.M)
    lines = [l.replace("\n", "") for prot in fasta_str.split(">") for l in prot.strip().split("\n", 1)][1:]
    tags, seqs = lines[::2], lines[1::2]
    tags = [re.split("\W| \|", t)[0] for t in tags]

    return tags, seqs


def update_timings(
    timing_dict: Dict[str, float],
    output_file: os.PathLike = os.path.join(os.getcwd(), "timings.json"),
):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)

    return output_file


def run_model(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    tag: str,
    output_dir: os.PathLike,
) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
        update_timings({tag: {"inference": inference_time}}, os.path.join(output_dir, "timings.json"))

    return out


# prepare_output

# relax_protein
