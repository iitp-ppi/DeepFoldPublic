# Copyright 2024 DeepFold Team


"""Runs MSA and template search and returns pickled features."""

import argparse
import gzip
import json
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

from deepfold.model.v2.data import pipeline
from deepfold.search import templates
from deepfold.search.tools import hmmsearch

logger = logging.getLogger(__name__)

MAX_TEMPLATE_HITS: int = 20


def main():
    parser = argparse.ArgumentParser(description="Runs MSA and template search and returns pickled features.")

    parser.add_argument("--fasta_path", required=True, help="Path to FASTA file.")
    parser.add_argument("--output_dir", required=True, help="Path to a directory that will store the results.")

    parser.add_argument(
        "--hhblits_binary_path",
        default=shutil.which("hhblits"),
        help="Path to the HHblits executable.",
    )
    parser.add_argument(
        "--jackhmmer_binary_path",
        default=shutil.which("jackhmmer"),
        help="Path to the JackHMMER executable.",
    )
    # parser.add_argument(
    #     "--hhsearch_binary_path",
    #     default=shutil.which("hhsearch"),
    #     help="Path to the HHsearch executable.",
    # )
    parser.add_argument(
        "--hmmsearch_binary_path",
        default=shutil.which("hmmsearch"),
        help="Path to the hmmsearch executable.",
    )
    parser.add_argument(
        "--hmmbuild_binary_path",
        default=shutil.which("hmmbuild"),
        help="Path to the hmmbuild executable.",
    )
    parser.add_argument(
        "--kalign_binary_path",
        default=shutil.which("kalign"),
        help="Path to the Kalign executable.",
    )

    parser.add_argument(
        "--uniref90_database_path",
        help="Path to the UniRef90 database.",
    )
    parser.add_argument(
        "--mgnify_database_path",
        help="Path to the MGnify database.",
    )
    parser.add_argument(
        "--bfd_database_path",
        help="Path to the BFD database.",
    )
    parser.add_argument(
        "--uniref30_database_path",
        help="Path to the UniRef30/UniClust30 database.",
    )
    parser.add_argument(
        "--uniprot_database_path",
        help="Path to the UniRef90 database.",
    )

    parser.add_argument(
        "--use_small_bfd",
        action="store_true",
        help="Use small BFD instead.",
    )

    parser.add_argument(
        "--pdb_seqres_database_path",
        help="Path to the PDB seqres database for use by hmmsearch..",
    )
    parser.add_argument(
        "--pdb_archive_dir",
        help="Path to a directroy with the PDB archive. Each file named '<pdb_id>.cif.gz'.",
    )
    parser.add_argument(
        "--max_template_date",
        help="Maximum template release date to consider.",
    )
    parser.add_argument(
        "--obsolete_pdbs_path",
        help="Path to file containing a mapping from obsolete PDB IDs to the PDB IDs of their replacements.",
    )

    parser.add_argument(
        "--use_precomputed_msas",
        action="store_true",
        help="Read MSA files in the output directory.",
    )

    parser.add_argument(
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPUs to compute alignments.",
    )

    args = parser.parse_args()
    for k, v in vars(args).items():
        logger.info(f"{k}={v}")

    generate_pkl_features(args)


def generate_pkl_features(args):
    """
    Generate pickled features for the given sequence.
    """

    logger.info(f"Search homogeneous sequences and structures for '{args.fasta_path}'")
    timings = {}
    output_path = Path(args.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    target_name = os.path.splitext(os.path.split(args.fasta_path)[-1])[0]

    # Get features
    features_output_path = output_path / f"features.pkl.gz"

    t_0 = time.time()

    template_searcher = hmmsearch.Hmmsearch(
        binary_path=args.hmmsearch_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        database_path=args.pdb_seqres_database_path,
    )

    template_featurizer = templates.HmmsearchHitFeaturizer(
        mmcif_dir=args.pdb_archive_dir,
        max_template_date=args.max_template_date,
        max_hits=MAX_TEMPLATE_HITS,
        kalign_binary_path=args.kalign_binary_path,
        obsolete_pdbs_path=args.obsolete_pdbs_path,
    )

    if not args.use_precomputed_msas:
        pipeline.AlignmentRunner(
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            bfd_database_path=args.bfd_database_path,
            uniref30_database_path=args.uniref30_database_path,
            uniprot_database_path=args.uniprot_database_path,
            template_searcher=template_searcher,
            use_small_bfd=args.use_small_bfd,
            num_cpus=args.num_cpus,
        ).run(args.fasta_path, args.output_dir)

    feature_dict = pipeline.DataPipeline(template_featurizer).process_fasta(args.fasta_path, args.output_dir)
    timings["features"] = time.time() - t_0

    pickle.dump(feature_dict, gzip.GzipFile(features_output_path, "wb"), protocol=4)

    logging.info(f"Final timings for '{target_name}': {timings}")

    timings_output_path = output_path / "timings.json"
    with open(timings_output_path, "w") as fp:
        fp.write(json.dumps(timings, indent=4))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
