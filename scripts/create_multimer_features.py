import argparse
import collections
import logging
import re
import string
import sys
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt

from deepfold.data.multimer.input_features import ComplexInfo, process_multimer_features
from deepfold.data.search.parsers import convert_stockholm_to_a3m
from deepfold.eval.plot import plot_msa
from deepfold.utils.file_utils import dump_pickle, load_pickle

logger = logging.getLogger(__name__)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, s):
        for fp in self.files:
            fp.write(s)

    def flush(self):
        for fp in self.files:
            fp.flush()


def parse_stoi(stoi_str: str):
    m = re.findall("[A-Z]([1-9][0-9]*)", stoi_str)
    return list(map(int, m))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_filepath",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--target_dirpath",
        default=Path.cwd(),
    )
    parser.add_argument(
        "-o",
        "--output_dirpath",
        default=Path.cwd(),
    )
    parser.add_argument(
        "-l",
        "--log_filepath",
        default=Path.cwd().joinpath("list.multimer"),
    )
    parser.add_argument(
        "-s",
        "--from",
        default=0,
        type=int,
        dest="start_num",
    )
    parser.add_argument(
        "--non_pair",
        action="store_true",
        dest="non_pair",
    )
    args = parser.parse_args()
    return args


def parse_note(note_str: str) -> Tuple[str, str, List[str]]:
    lines = [s.partition("#")[0].strip() for s in note_str.strip().splitlines()]
    lines = list(filter(lambda s: bool(s), lines))
    target_id, stoi = lines[0].split()

    recipes = []
    for line in lines[1:]:
        recipes.append(line.split()[1:])

    return target_id, stoi, recipes


def process_uniprot(
    num_chains: int,
    target_id: str,
    target_dirpath: Path,
    output_dirpath: Path,
):
    a3m_strings = {}
    for i in range(num_chains):
        chain_id = f"T{target_id[1:]}s{i+1}"
        output_filepath = output_dirpath.joinpath(f"msas/{chain_id}.uniprot.a3m")
        if output_filepath.exists():
            with open(output_filepath, "r") as fp:
                a3m_strings[chain_id] = fp.read()
        else:
            with open(target_dirpath / f"{chain_id}/msas/uniprot_hits.sto", "r") as fp:
                a3m_str = convert_stockholm_to_a3m(fp.read(), max_sequences=50000)
            if not output_filepath.parent.exists():
                output_filepath.parent.mkdir(parents=True)
            with open(output_filepath, "w") as fp:
                fp.write(a3m_str)
            a3m_strings[chain_id] = a3m_str

    return a3m_strings


def main(args: argparse.Namespace):
    # Parse recipes:
    with open(args.input_filepath, "r") as fp:
        note_str = fp.read()
    target_id, stoichiom, recipes = parse_note(note_str)
    cardinality = parse_stoi(stoichiom)
    chain_ids = [f"T{target_id[1:]}s{i}" for i in range(1, len(cardinality) + 1)]

    # Process UniProt:
    a3m_strings = process_uniprot(
        num_chains=len(cardinality),
        target_id=target_id,
        target_dirpath=args.target_dirpath,
        output_dirpath=args.output_dirpath,
    )

    with open(args.log_filepath, "a") as fp:
        tee = Tee(sys.stdout, fp)

        tee.write(f"#\n")

        for i, pair in enumerate(recipes, start=args.start_num):
            feats = {cid: load_pickle(args.target_dirpath / f"{cid}/{y}/features.pkz") for cid, y in zip(chain_ids, pair)}
            name = "".join(f"{a}{n}" for a, n in zip(string.ascii_uppercase, cardinality) if n > 0)
            name += f"_{i}"

            in_cid = [cid for cid, n in zip(chain_ids, cardinality) if n > 0]
            in_car = [n for n in cardinality if n > 0]

            complex = ComplexInfo(in_cid, in_car)
            example = process_multimer_features(
                complex=complex,
                all_monomer_features=feats,
                # ,
                a3m_strings_with_identifiers=(collections.defaultdict(str) if args.non_pair else a3m_strings),
            )
            out_path = args.output_dirpath.joinpath(f"{name}/features.pkz")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            dump_pickle(example, out_path)

            fig = plot_msa(example)
            # plt.show(fig)
            fig.savefig(args.output_dirpath.joinpath(f"{name}/msa_depth.png"))
            plt.close(fig)

            tee.write(f"{name:12s} ")
            tee.write(" ".join(f"{p:3s}" if n > 0 else "-" for p, n in zip(pair, cardinality)))
            tee.write("\n")
            tee.flush()


if __name__ == "__main__":
    main(parse_args())
