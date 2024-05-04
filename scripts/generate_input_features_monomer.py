import argparse
import logging
import os
from pathlib import Path

from deepfold.data import data_pipeline, templates
from deepfold.utils.file_utils import dump_pickle
from deepfold.utils.script_utils import parse_fasta
from scripts.utils import add_data_args_

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


MAX_TEMPLATE_HITS = 20


def generate_feature_dict(tags, seqs, alignment_dir, data_processor) -> dict:
    if len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        fasta_string = f">{tag}\n{seq}"

        feature_dict = data_processor.process_fasta_string(
            fasta_string=fasta_string,
            alignment_dir=alignment_dir,
        )
    else:
        raise NotImplementedError("Don't support multi-sequencial FASTA files")

    return feature_dict


def main(args: argparse.Namespace) -> None:
    # Make output directory:
    args.output_dirpath.mkdir(parents=True, exist_ok=True)

    # Template featurizer:
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=args.mmcif_dirpath,
        max_template_date=args.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=args.kalign_binary_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
    )

    # Input feature processor:
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    # Parse a FASTA file:
    logger.info(f"Load FASTA file '{args.fasta_filepath}'")
    with open(args.fasta_filepath, "r") as fp:
        fasta_str = fp.read()
    tags, seqs = parse_fasta(fasta_str)

    # Generate input features:
    feats = generate_feature_dict(tags, seqs, args.alignment_dirpath, data_processor)

    # Dump to pickle
    pkl_filepath = args.output_dirpath / "features.pkz"
    logger.info(f"Save input features on '{pkl_filepath}'")
    dump_pickle(feats, pkl_filepath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_filepath",
        type=Path,
    )
    parser.add_argument(
        "alignment_dirpath",
        type=Path,
    )
    parser.add_argument(
        "output_dirpath",
        type=Path,
    )
    parser.add_argument(
        "--mmcif_dirpath",
        type=Path,
        required=True,
    )
    add_data_args_(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
    exit(0)
