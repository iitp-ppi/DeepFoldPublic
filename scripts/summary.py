#!/usr/bin/env python3

import argparse
import json
import math
from glob import glob
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "-k",
        "--key",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-r",
        "--reverse",
        action="store_true",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    records = []
    paths = sum([[Path(p) for p in glob(path)] for path in args.paths], [])
    for path in paths:
        rec = []
        with open(path, "r") as fp:
            obj = json.load(fp)
        model_name = str(path.parent) + "/" + obj["model_name"]
        suffix = obj.get("suffix", "")
        rec.append(model_name + suffix)
        rec.append(sum(obj["plddt"]) / len(obj["plddt"]))
        rec.append(obj.get("ptm", 0.0))
        rec.append(obj.get("iptm", 0.0))
        rec.append(obj.get("weighted_ptm_score", 0.0))
        records.append(rec)

    if len(paths) == 0:
        print("No summary found...")
        exit(0)

    rev = args.key > 1
    if args.reverse:
        rev = not rev
    records.sort(key=lambda x: x[args.key - 1], reverse=rev)

    len_name = max(map(lambda x: len(x[0]), records))
    len_pad = int(math.log10(len(records)) + 1)
    lpad = " " * (len_pad + 1)

    header = lpad + "Model".ljust(len_name) + "\tplDDT\tpTM\tipTM\tWeighted TM"
    print(header)
    for i, rec in enumerate(records, start=1):
        num = f"{i}".rjust(len_pad) + " "
        print(
            num
            + "{0:s}\t{1:2.2f}\t{2:0.4f}\t{3:0.4f}\t{4:0.4f}".format(
                rec[0].ljust(len_name),
                *rec[1:],
            )
        )


if __name__ == "__main__":
    main(parse_args())
