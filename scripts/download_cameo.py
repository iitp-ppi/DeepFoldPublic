# Copyright 2023 DeepFold Team

import argparse
import json
import logging
import os
import re

import requests

from deepfold.data import mmcif_parsing
from deepfold.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)

VALID_PERIODS = [
    "1-year",
    "6-months",
    "3-months",
    "1-month",
    "1-week",
]

DATE_REGEX = re.compile("^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


def generate_url(period: str, end_date: str) -> str:
    return f"https://www.cameo3d.org/modeling/targets/{period}/ajax/?to_date={end_date}"


def download_cameo(args):
    data_dir_path = os.path.join(args.output_dir, "data_dir")
    fasta_dir_path = os.path.join(args.output_dir, "fasta_dir")
    cameo_dump_path = os.path.join(args.output_dir, "cameo.json")

    os.makedirs(data_dir_path, exist_ok=True)
    os.makedirs(fasta_dir_path, exist_ok=True)

    url = generate_url(args.period, args.end_date)
    raw_data = requests.get(url).text
    parsed_data = json.loads(raw_data)

    with open(cameo_dump_path, "w") as fp:
        json.dump(parsed_data, fp, indent=4)

    chain_data = parsed_data["aaData"]
    for chain in chain_data:
        rcsb_id = chain["pdbid"]
        chain_id = chain["pdbid_chain"]

        cif_url = f"https://files.rcsb.org/view/{rcsb_id.upper()}.cif"
        logger.info(f"GET [{rcsb_id}:{chain_id}] {cif_url}")
        cif_str = requests.get(cif_url).text

        parsed_cif = mmcif_parsing.parse(file_id=rcsb_id, mmcif_string=cif_str)
        mmcif_obj = parsed_cif.mmcif_object
        if mmcif_obj is None:
            raise list(parsed_cif.errors.values())[0]

        seq = mmcif_obj.chain_to_seqres[chain_id]

        fasta_str = "\n".join([f">{rcsb_id}_{chain_id}", seq])

        fasta_filename = f"{rcsb_id}_{chain_id}.fasta"
        with open(os.path.join(fasta_dir_path, fasta_filename), "w") as fp:
            fp.write(fasta_str)

        cif_filename = f"{rcsb_id}.cif"
        with open(os.path.join(data_dir_path, cif_filename), "w") as fp:
            fp.write(cif_str)


def main():
    setup_logging("fetch.log")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "period",
        type=str,
        help=f"The length of the period to download CAMEO proteins. Choose from {VALID_PERIODS}",
    )
    parser.add_argument(
        "end_date",
        type=str,
        help="The date marking the end of the period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default=os.getcwd(),
        help="The directory to download CAMEO proteins.",
    )

    args = parser.parse_args()

    if args.period not in VALID_PERIODS:
        raise ValueError(f"Invalid period")

    if not DATE_REGEX.match(args.end_date):
        raise ValueError(f"Invalid end_date: {args.end_date}")

    logger.info(f"Get CAMEO target of {args.period} until {args.end_date}")
    download_cameo(args)


if __name__ == "__main__":
    main()
