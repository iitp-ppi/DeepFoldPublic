import argparse
import collections
import logging
import os
import re
import string
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib import pyplot as plt

from deepfold.data.multimer.input_features import ComplexInfo, process_multimer_features
from deepfold.data.search.parsers import convert_stockholm_to_a3m
from deepfold.eval.plot import plot_msa
from deepfold.utils.file_utils import dump_pickle, get_file_content_and_extension, load_pickle

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
    m = re.findall("[A-Z]([0-9]*)", stoi_str)
    return list(map(int, m))


def parse_args() -> argparse.Namespace:

    pairing_preset = ["none", "uniprot", "colab", "taxid"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_filepath",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-d",
        "--target_dirpath",
        default=Path.cwd(),
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output_dirpath",
        default=Path.cwd(),
        type=Path,
    )
    parser.add_argument(
        "-l",
        "--log_filepath",
        default=Path.cwd().joinpath("list.multimer"),
        type=Path,
    )
    parser.add_argument(
        "-n",
        "--start_num",
        default=0,
        type=int,
        dest="start_num",
    )
    parser.add_argument(
        "--pairing",
        default="uniprot",
        choices=pairing_preset,
    )
    parser.add_argument(
        "--colab_a3m",
        type=Path,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files.",
    )
    parser.add_argument(
        "--casp",
        action="store_true",
        help="Apply the CASP name converntion.",
    )
    args = parser.parse_args()

    return args


def parse_note(note_str: str) -> Tuple[str, str, str | None, List[str]]:
    lines = [s.partition("#")[0].strip() for s in note_str.strip().splitlines()]
    lines = list(filter(lambda s: bool(s), lines))
    ls = lines[0].split()
    target_id = ls[0]
    stoi = ls[1]
    if len(ls) > 2:
        suffix = ls[2]
    else:
        suffix = None

    recipes = []
    for line in lines[1:]:
        recipes.append(line.split()[1:])

    return target_id, stoi, suffix, recipes


def process_uniprot(
    chain_ids: List[str],
    target_dirpath: Path,
    output_dirpath: Path,
):
    a3m_strings = {}
    for chain_id in chain_ids:
        # chain_id = f"T{target_id[1:]}s{i+1}"
        output_filepath = output_dirpath.joinpath(f"msas/{chain_id}.uniprot.a3m")
        if output_filepath.exists():
            with open(output_filepath, "r") as fp:
                a3m_strings[chain_id] = get_file_content_and_extension(output_filepath)[0]
        else:
            sto_str, suffix = get_file_content_and_extension(target_dirpath / f"{chain_id}/msas/uniprot_hits.sto")
            assert suffix == ".sto"
            a3m_str = convert_stockholm_to_a3m(sto_str, max_sequences=50000)
            if not output_filepath.parent.exists():
                output_filepath.parent.mkdir(parents=True)
            with open(output_filepath, "w") as fp:
                fp.write(a3m_str)
            a3m_strings[chain_id] = a3m_str

    return a3m_strings


def unserialize_msa(a3m_lines: List[str]) -> Tuple[
    Optional[List[str]],
    Optional[List[str]],
    List[str],
    List[int],
]:
    a3m_lines = a3m_lines[0].replace("\x00", "").splitlines()
    assert a3m_lines[0].startswith("#") and len(a3m_lines[0][1:].split("\t")) == 2

    if len(a3m_lines) < 3:
        raise ValueError(f"Unknown file format a3m")
    tab_sep_entries = a3m_lines[0][1:].split("\t")
    query_seq_len = tab_sep_entries[0].split(",")
    query_seq_len = list(map(int, query_seq_len))
    query_seqs_cardinality = tab_sep_entries[1].split(",")
    query_seqs_cardinality = list(map(int, query_seqs_cardinality))
    is_homooligomer = True if len(query_seq_len) == 1 and query_seqs_cardinality[0] > 1 else False
    is_single_protein = True if len(query_seq_len) == 1 and query_seqs_cardinality[0] == 1 else False
    query_seqs_unique = []
    prev_query_start = 0
    # we store the a3m with cardinality of 1
    for n, query_len in enumerate(query_seq_len):
        query_seqs_unique.append(a3m_lines[2][prev_query_start : prev_query_start + query_len])
        prev_query_start += query_len
    paired_msa = [""] * len(query_seq_len)
    unpaired_msa = [""] * len(query_seq_len)
    already_in = dict()
    for i in range(1, len(a3m_lines), 2):
        header = a3m_lines[i]
        seq = a3m_lines[i + 1]
        if (header, seq) in already_in:
            continue
        already_in[(header, seq)] = 1
        has_amino_acid = [False] * len(query_seq_len)
        seqs_line = []
        prev_pos = 0
        for n, query_len in enumerate(query_seq_len):
            paired_seq = ""
            curr_seq_len = 0
            for pos in range(prev_pos, len(seq)):
                if curr_seq_len == query_len:
                    prev_pos = pos
                    break
                paired_seq += seq[pos]
                if seq[pos].islower():
                    continue
                if seq[pos] != "-":
                    has_amino_acid[n] = True
                curr_seq_len += 1
            seqs_line.append(paired_seq)

        # if sequence is paired add them to output
        if not is_single_protein and not is_homooligomer and sum(has_amino_acid) > 1:  # at least 2 sequences are paired
            header_no_faster = header.replace(">", "")
            header_no_faster_split = header_no_faster.split("\t")
            for j in range(0, len(seqs_line)):
                paired_msa[j] += ">" + header_no_faster_split[j] + "\n"
                paired_msa[j] += seqs_line[j] + "\n"
        else:
            for j, seq in enumerate(seqs_line):
                if has_amino_acid[j]:
                    unpaired_msa[j] += header + "\n"
                    unpaired_msa[j] += seq + "\n"
    if is_homooligomer:
        # homooligomers
        num = 101
        paired_msa = [""] * query_seqs_cardinality[0]
        for i in range(0, query_seqs_cardinality[0]):
            paired_msa[i] = ">" + str(num + i) + "\n" + query_seqs_unique[0] + "\n"
    if is_single_protein:
        paired_msa = None

    return (
        unpaired_msa,
        paired_msa,
        query_seqs_unique,
        query_seqs_cardinality,
    )


def main(args: argparse.Namespace):
    # Parse recipes:
    note_str, _ = get_file_content_and_extension(args.input_filepath)
    target_id, stoichiom, suffix, recipes = parse_note(note_str)
    cardinality = parse_stoi(stoichiom)
    if args.casp:
        chain_ids = [f"T{target_id[1:]}s{i}" for i in range(1, len(cardinality) + 1)]
    else:
        chain_ids = [f"{target_id}s{i}" for i in range(1, len(cardinality) + 1)]

    a3m_strings_with_identifiers = collections.defaultdict(str)
    paired_a3m_strings = dict()
    if args.pairing == "uniprot":  # Process UniProt:
        paired_a3m_strings = process_uniprot(
            chain_ids=chain_ids,
            target_dirpath=args.target_dirpath,
            output_dirpath=args.output_dirpath,
        )
    elif args.pairing == "colab":
        a3m_lines, _ = get_file_content_and_extension(args.colab_a3m)
        _, paired_msa, query_seqs_unique, _ = unserialize_msa([a3m_lines])
        assert paired_msa  # Check paired MSA is not None.
        paired_a3m_strings = {k: v for k, v in zip(chain_ids, paired_msa)}
        assert len(query_seqs_unique) == len(cardinality)
    elif args.pairing == "taxid":
        paired_a3m_strings = {}
        a3m_strings_with_identifiers = None
    elif args.pairing == "none":
        pass
    else:
        raise ValueError(f"Wrong pairing strategy: '{args.pairing}'")

    with open(args.log_filepath, "a") as fp:
        tee = Tee(sys.stdout, fp)

        tee.write(f"#\n")

        for i, pair in enumerate(recipes, start=args.start_num):
            name = "".join(f"{a}{n}" for a, n in zip(string.ascii_uppercase, cardinality) if n > 0)
            if suffix:
                name += f"_{suffix}"
            name += f"_{i}"
            out_path = args.output_dirpath.joinpath(f"{name}/features.pkz")
            if not args.force and out_path.exists():
                logger.error(f"PASS! Output file exists: {str(out_path)}")
                logger.info(f"Use --force to overwrite.")
                continue

            feats = {}
            for cid, y in zip(chain_ids, pair):
                ys = y.split("/")
                assert len(ys) >= 1
                if len(ys) == 1:
                    feat_filepath = args.target_dirpath / f"{cid}/{y}/features.pkz"
                else:
                    if os.path.splitext(y)[1] == ".pkz":
                        feat_filepath = args.target_dirpath.joinpath(y)
                    else:
                        feat_filepath = args.target_dirpath / f"{y}/features.pkz"
                feat = load_pickle(feat_filepath)
                feats[cid] = feat

            in_cid = [cid for cid, n in zip(chain_ids, cardinality) if n > 0]
            in_car = [n for n in cardinality if n > 0]

            complex = ComplexInfo(in_cid, in_car)
            example = process_multimer_features(
                complex=complex,
                all_monomer_features=feats,
                paired_a3m_strings=paired_a3m_strings,
                a3m_strings_with_identifiers=a3m_strings_with_identifiers,
                pair_with_identifier=(args.pairing == "taxid"),
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            dump_pickle(example, out_path)

            fig = plot_msa(example)
            # plt.show(fig)
            fig.savefig(args.output_dirpath.joinpath(f"{name}/msa_depth.png"))
            plt.close(fig)

            tee.write(f"{name:12s} ")
            tee.write(" ".join(f"{s:^3s}" if n > 0 else "-" for s, n in zip(pair, cardinality)))
            tee.write(f"  # pair={args.pairing}")
            tee.write("\n")
            tee.flush()


if __name__ == "__main__":
    main(parse_args())
