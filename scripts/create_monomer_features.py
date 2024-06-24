import argparse
import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np

from deepfold.common import residue_constants as rc
from deepfold.data.search.crfalign import parse_crf
from deepfold.data.search.input_features import create_msa_features, create_sequence_features, create_template_features
from deepfold.data.search.parsers import convert_stockholm_to_a3m, parse_fasta, parse_hhr, parse_hmmsearch_sto
from deepfold.data.search.templates import TemplateHitFeaturizer, create_empty_template_feats
from deepfold.utils.file_utils import dump_pickle, load_pickle
from deepfold.utils.log_utils import setup_logging

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Domain:
    doi: int
    start: int
    end: int
    model_name: str


def parse_dom(dom_str: str) -> Tuple[List[Domain], List[str]]:
    dom_str = dom_str.strip()
    lines = [s.partition("#")[0].strip() for s in dom_str.strip().splitlines()]
    lines = list(filter(lambda s: bool(s), lines))

    crf_codes = []
    domains = []
    for line in lines:
        ls = line.split()
        doi = int(ls[0])
        if doi == 0:
            crf_codes.extend(ls[1:])
        else:
            assert len(ls) == 3
            start, end = map(int, ls[1].split("-"))
            start -= 1
            name = ls[2]
            domain = Domain(doi=doi, start=start, end=end, model_name=name)
            domains.append(domain)

    return domains, crf_codes


def get_domains(
    domains: List[Domain],
    query_name: str,
    query_seq: str,
) -> dict:
    template_domain_names = []
    template_sequence = []
    template_aatype = []
    template_all_atom_positions = []
    template_all_atom_mask = []
    template_sum_probs = []

    seqlen = len(query_seq)
    aatype = rc.sequence_to_onehot(query_seq, rc.HHBLITS_AA_TO_ID)

    for dom in domains:
        ns = dom.model_name.split("/")
        path = "/".join([f"{query_name}_{dom.doi}"] + ns[:-1] + [f"result_{ns[-1]}.pkz"])
        results = load_pickle(path)

        pos = np.pad(results["final_atom_positions"], ((dom.start, seqlen - dom.end), (0, 0), (0, 0)))
        mask = np.pad(results["final_atom_mask"], ((dom.start, seqlen - dom.end), (0, 0)))

        print(">", f"[{dom.doi}]", dom.model_name, f"[{dom.start}, {dom.end})", "->", pos.shape, mask.shape)

        template_sum_probs.append(1.0)
        template_domain_names.append(dom.model_name.encode())
        template_sequence.append(query_seq.encode())
        template_aatype.append(aatype.copy())
        template_all_atom_positions.append(pos)
        template_all_atom_mask.append(mask)

    template_features = {
        "template_domain_names": np.array(template_domain_names, dtype=np.object_),
        "template_sequence": np.array(template_sequence, dtype=np.object_),
        "template_aatype": np.array(template_aatype, dtype=np.int32),
        "template_all_atom_positions": np.stack(template_all_atom_positions, dtype=np.float32),
        "template_all_atom_mask": np.stack(template_all_atom_mask, dtype=np.float32),
        "template_sum_probs": np.array(template_sum_probs, dtype=np.float32).reshape([-1, 1]),
    }

    return template_features


def main(args: argparse.Namespace) -> None:
    # Parse input FASTA file:
    with open(args.fasta_filepath, "r") as fp:
        fasta_str = fp.read().strip()
    sequences, descriptions = parse_fasta(fasta_str)
    assert len(sequences) == len(descriptions) == 1

    query_sequence = sequences[0]
    description = descriptions[0]
    seqlen = len(query_sequence)

    logger.info("Input FASTA path: {}".format(args.fasta_filepath))
    logger.info("Description: {}".format(description))
    logger.info("Sequence: {}".format(query_sequence))

    domain_name = description.split()[0]
    sequence_features = create_sequence_features(query_sequence, domain_name)
    if args.offset:
        logger.info(f"Shift residue index for {args.offset}")
        sequence_features["residue_index"] += args.offset

    # Featurize template hits:
    template_features = create_empty_template_feats(seqlen)
    additional_template_features = None
    if args.template_filepath is not None:
        logger.info("Prepare template featurzer...")
        logger.info("PDB mmCIF directory path: {}".format(args.pdb_mmcif_dirpath))
        logger.info("PDB mmCIF obsolates list: {}".format(args.pdb_obsolete_filepath))

        template_featurizer = TemplateHitFeaturizer(
            max_template_hits=args.max_template_hits,
            pdb_mmcif_dirpath=args.pdb_mmcif_dirpath,
            kalign_executable_path=args.kalign_binary_path,
            verbose=True,
        )

        logger.info("Parse {}".format(args.template_filepath))
        suffix = args.template_filepath.suffix
        with open(args.template_filepath, "r") as fp:
            template_str = fp.read()

        sort_by_sum_probs = True
        mode = None

        if args.template_mode == "auto":
            if suffix == ".sto":
                mode = "hhm"
            elif suffix == ".hhr":
                mode = "hhr"
            elif suffix == ".crf":
                mode = "crf"
            else:
                raise RuntimeError(f"Not supported template hits extensions: {suffix}")
        else:
            mode = args.template_mode

        template_hits = []
        if mode == "hhm":
            template_hits = parse_hmmsearch_sto(query_sequence, template_str)
        elif mode == "hhr":
            template_hits = parse_hhr(template_str)
        elif mode == "crf":
            if args.crf_alignment_dirpath is None:
                raise ValueError("CRFalign hits are provided. However, args.crf_alignment_dirpath is None")
            else:
                template_hits = parse_crf(
                    template_str,
                    query_id=domain_name,
                    alignment_dir=args.crf_alignment_dirpath,
                )
                sort_by_sum_probs = False
        elif mode == "dom":
            domains, crf_chains = parse_dom(template_str)
            template_features = create_empty_template_feats(seqlen, empty=True)
            if crf_chains:
                template_hits = parse_crf(
                    "\n".join(crf_chains),
                    query_id=domain_name,
                    alignment_dir=args.crf_alignment_dirpath,
                )
            sort_by_sum_probs = False
        else:
            raise RuntimeError(f"Not supported template mode: {suffix}")

        if template_hits:
            template_features = create_template_features(
                sequence=query_sequence,
                template_hits=template_hits,
                template_hit_featurizer=template_featurizer,
                max_release_date=args.max_template_date,
                sort_by_sum_probs=sort_by_sum_probs,
                # shuffling_seed=args.seed,
            )

    if domains:
        additional_template_features = get_domains(domains, domain_name, query_sequence)

    if additional_template_features:
        for k, v in template_features.items():
            print(">>", k, v.dtype, v.shape)
            template_features[k] = np.concatenate([v, additional_template_features[k]], axis=0)

    # Create MSA features:
    a3m_strings = []
    max_num_seqs = {
        "bfd_uniclust_hits": None,
        "mgnify_hits": 5000,
        "uniref90_hits": 10000,
        "uniprot_hits": 50000,
    }
    if args.alignment_filepaths is not None:
        logger.info("Parse MSA search results...")
        for path in args.alignment_filepaths:
            logger.info("Parse {}".format(path))
            with open(path, "r") as fp:
                if path.suffix == ".a3m":
                    a3m_str = fp.read()
                elif path.suffix == ".sto":
                    a3m_str = convert_stockholm_to_a3m(fp.read(), max_sequences=max_num_seqs.get(path.stem, None))
                else:
                    raise RuntimeError(f"Not supported MSA search extensions: {suffix}")
            a3m_strings.append(a3m_str)
    msa_features = create_msa_features(
        a3m_strings,
        sequence=query_sequence,
    )

    logger.info("Write input features on {}".format(args.output_filepath))
    args.output_filepath.parent.mkdir(parents=True, exist_ok=True)
    input_features = {**sequence_features, **msa_features, **template_features}
    dump_pickle(input_features, args.output_filepath, level=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fasta_filepath",
        type=Path,
        required=True,
        help="Input FASTA file path.",
    )
    parser.add_argument(
        "-a",
        "--alignment_filepaths",
        action="append",
        type=Path,
        help="MSA search result files.",
    )
    parser.add_argument(
        "-t",
        "--template_filepath",
        type=Path,
        default=None,
        help="Template search result files.",
    )
    parser.add_argument(
        "-o",
        "--output_filepath",
        type=Path,
        required=True,
        help="Output compressed features file path.",
    )
    parser.add_argument(
        "--pdb_mmcif_dirpath",
        type=Path,
        default=None,
        help="PDB gzipped mmCIF files directory.",
    )
    parser.add_argument(
        "--pdb_obsolete_filepath",
        type=Path,
        default=None,
        help="PDB obsoleted entry list file.",
    )
    parser.add_argument(
        "--max_template_date",
        type=str,
        default=datetime.today().strftime(r"%Y-%m-%d"),
        help="Maximum template release date.",
    )
    parser.add_argument(
        "--max_template_hits",
        type=int,
        default=20,
        help="Maximum template hits.",
    )
    parser.add_argument(
        "--template_mode",
        type=str,
        default="auto",
        help="Template mode.",
    )
    parser.add_argument(
        "--crf_alignment_dirpath",
        type=Path,
        default=None,
        help="CRFalign alignment directory.",
    )
    parser.add_argument(
        "--kalign_binary_path",
        type=str,
        default="kalign",
        help="Kalign binary filepath.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Shuffling seeds for the template featurizer.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for the residue numbers.",
    )
    args = parser.parse_args()

    assert args.template_mode in ["auto", "hhr", "hhm", "crf", "dom"]

    setup_logging("features.log", mode="a")

    for k, v in vars(args).items():
        if isinstance(v, list):
            for i, x in enumerate(v):
                print(f"{k}[{i}]={x}")
        else:
            print(f"{k}={v}")

    return args


if __name__ == "__main__":
    main(parse_args())
    exit(0)
