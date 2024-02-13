"""Genetic search."""

import argparse
import datetime
import gzip
import logging
import os
import pickle

from deepfold.model.v2.data.pipeline import DataPipeline
from deepfold.model.v2.data.search import AlignmentRunner
from deepfold.model.v2.search.utils import SchemeRegularizer
from deepfold.search.templates import HmmsearchHitFeaturizer
from deepfold.search.tools import hmmsearch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_TEMPLATE_HITS = 20


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--fasta_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="Path to alignment directory. If provided, alignment computation is skipped",
    )
    parser.add_argument("--compress", type=bool, action="store_true")

    parser.add_argument("--mmcif_dir", type=str, default=None)
    parser.add_argument("--uniref90_database_path", type=str, default=None)
    parser.add_argument("--mgnify_database_path", type=str, default=None)
    parser.add_argument("--uniref30_database_path", type=str, default=None)
    parser.add_argument("--bfd_database_path", type=str, default=None)
    parser.add_argument("--uniprot_database_path", type=str, default=None)
    parser.add_argument("--pdb_seqres_database_path", type=str, default=None)
    parser.add_argument("--jackhmmer_binary_path", type=str, default="jackhmmer")
    parser.add_argument("--hhblits_binary_path", type=str, default="hhblits")
    parser.add_argument("--hmmsearch_binary_path", type=str, default="hhsearch")
    parser.add_argument("--hmmbuild_binary_path", type=str, default="hmmbuild")
    parser.add_argument("--kalign_binary_path", type=str, default="kalign")
    parser.add_argument("--max_template_date", type=str, default=datetime.date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)

    args = parser.parse_args()

    if args.use_precomputed_alignments is None:
        logger.info(f"Run alignment with '{args.fasta_path}'")
    else:
        logger.info(f"Use precomputed alignments in '{args.use_precomputed_alignments}'")
    logger.info(f"Output results are saved to '{args.output_dir}'")

    omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
    num_cpus = int(omp_num_threads) if omp_num_threads is not None else None

    template_searcher = hmmsearch.Hmmsearch(
        binary_path=args.hmmsearch_binary_path,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        database_path=args.pdb_seqres_database_path,
    )

    if args.use_precomputed_alignments is None:
        alignment_runner = AlignmentRunner(
            template_searcher=template_searcher,
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            uniref30_database_path=args.unireft30_database_path,
            bfd_database_path=args.bfd_database_path,
            uniprot_database_path=args.uniprot_database_path,
            num_cpus=num_cpus,
        )
        alignment_runner.run(args.fasta_path, args.output_dir)

    if args.mmcif_dir is None:
        template_featurizer = None
    else:
        template_featurizer = HmmsearchHitFeaturizer(
            mmcif_dir=args.pdb_archive_dir,
            max_template_date=args.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=args.kalign_binary_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )

    data_processor = DataPipeline(template_featurizer)
    alignment_scheme = {
        "template": ["template_hits.sto"],
        "msa": ["*.sto", "*.a3m"],
    }
    alignment_scheme = SchemeRegularizer(
        ["template", "msa"],
        base_dir=args.output_dir,
    ).process(alignment_scheme)
    feature_dict = data_processor.process_fasta(
        args.fasta_path,
        args.output_dir,
        alignment_scheme=alignment_scheme,
    )

    output_path = os.path.join(args.output_dir, "features.pkl")
    if args.compress:
        output_path += ".gz"
    logger.info(f"Save generated features to '{output_path}'")

    if args.compress:
        with gzip.open(output_path, "wb") as fp:
            pickle.dump(feature_dict, fp)
    else:
        with open(output_path, "wb") as fp:
            pickle.dump(feature_dict, fp)


if __name__ == "__main__":
    main()
