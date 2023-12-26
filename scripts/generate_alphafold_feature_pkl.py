import argparse
import logging
import os
import pickle
from datetime import date

from deepfold.model.alphafold.data import templates
from deepfold.model.alphafold.data.pipeline_monomer import AlignmentRunner, DataPipeline

logger_cfg = {
    "format": r"%(asctime)s:%(name)s:%(levelname)s:%(message)s",
    "datefmt": r"%Y-%m-%d %H:%M:%S",
}
logging.basicConfig(level=logging.INFO, **logger_cfg)
logger = logging.getLogger(__name__)


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

    parser.add_argument("--mmcif_dir", type=str, default=None)
    parser.add_argument("--uniref90_database_path", type=str, default=None)
    parser.add_argument("--mgnify_database_path", type=str, default=None)
    parser.add_argument("--pdb70_database_path", type=str, default=None)
    parser.add_argument("--uniclust30_database_path", type=str, default=None)
    parser.add_argument("--bfd_database_path", type=str, default=None)
    parser.add_argument("--jackhmmer_binary_path", type=str, default="jackhmmer")
    parser.add_argument("--hhblits_binary_path", type=str, default="hhblits")
    parser.add_argument("--hhsearch_binary_path", type=str, default="hhsearch")
    parser.add_argument("--kalign_binary_path", type=str, default="kalign")
    parser.add_argument("--max_template_date", type=str, default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)
    parser.add_argument("--release_dates_path", type=str, default=None)

    args = parser.parse_args()

    if args.use_precomputed_alignments is None:
        logger.info(f"Run alignment with '{args.fasta_path}'")
    else:
        logger.info(f"Use precomputed alignments in '{args.use_precomputed_alignments}'")
    logger.info(f"Output results are saved to '{args.output_dir}")

    omp_num_threads = os.environ.get("OMP_NUM_THREADS", None)
    num_cpus = int(omp_num_threads) if omp_num_threads is not None else None

    if args.use_precomputed_alignments is None:
        alignment_runner = AlignmentRunner(
            uniref90_database_path=args.uniref90_database_path,
            mgnify_database_path=args.mgnify_database_path,
            pdb70_database_path=args.pdb70_database_path,
            uniclust30_database_path=args.uniclust30_database_path,
            bfd_database_path=args.bfd_database_path,
            jackhmmer_binary_path=args.jackhmmer_binary_path,
            hhblits_binary_path=args.hhblits_binary_path,
            hhsearch_binary_path=args.hhsearch_binary_path,
            num_cpus=num_cpus,
        )
        alignment_runner.run(args.fasta_path, args.output_dir)

    if args.mmcif_dir is None:
        template_featurizer = None
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=20,
            kalign_binary_path=args.kalign_binary_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
            release_dates_path=None,
        )

    data_processor = DataPipeline(template_featurizer)

    feature_dict = data_processor.process_fasta(args.fasta_path, args.output_dir)

    feature_pkl_path = os.path.join(args.output_dir, "features.pkl")
    logger.info(f"Save generated features to '{feature_pkl_path}'")
    with open(feature_pkl_path, "wb") as fp:
        pickle.dump(feature_dict, fp)


if __name__ == "__main__":
    main()
