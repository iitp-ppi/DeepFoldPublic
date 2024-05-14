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
    tags = [re.split(r"\W| \|", t)[0] for t in tags]

    return tags, seqs
