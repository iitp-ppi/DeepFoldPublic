import argparse
import logging
import os
from typing import Sequence

from deepfold.data import data_pipeline, templates
from deepfold.data.tools import hhsearch, hmmsearch
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


def precompute_alignments(
    tags: Sequence[str],
    seqs: Sequence[str],
    alignment_dir: os.PathLike,
    args: argparse.Namespace,
):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)

        if args.use_precomputed_alignments is None:
            logger.info(f"Generating alignments for '{tag}'")

            os.makedirs(local_alignment_dir, exist_ok=True)

            if "multimer" in args.config_preset:
                template_searcher = hmmsearch.Hmmsearch(
                    binary_path=args.hmmsearch_binary_path,
                    hmmbuild_binary_path=args.hmmbuild_binary_path,
                    database_path=args.pdb_seqres_database_path,
                )
            else:
                template_searcher = hhsearch.HHSearch(
                    binary_path=args.hhsearch_binary_path,
                    databases=[args.pdb70_database_path],
                )

            alignment_runner = data_pipeline.AlignmentRunner(
                jackhmmer_binary_path=args.jackhmmer_binary_path,
                hhblits_binary_path=args.hhblits_binary_path,
                uniref90_database_path=args.uniref90_database_path,
                mgnify_database_path=args.mgnify_database_path,
                bfd_database_path=args.bfd_database_path,
                uniref30_database_path=args.uniref30_database_path,
                uniclust30_database_path=args.uniclust30_database_path,
                uniprot_database_path=args.uniprot_database_path,
                template_searcher=template_searcher,
                use_small_bfd=args.bfd_database_path is None,
                num_cpus=args.cpus,
            )

            alignment_runner.run(tmp_fasta_path, local_alignment_dir)
        else:
            logger.info(f"Using precomputed alignments for '{tag}' at '{alignment_dir}'")

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    if args.is_multimer:
        feature_dict = data_processor.process_fasta_string(
            fasta_string="\n".join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]),
            alignment_dir=alignment_dir,
            non_pair=args.non_pair,
        )
    elif len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        fasta_string = f">{tag}\n{seq}"

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta_string(
            fasta_string=fasta_string,
            alignment_dir=local_alignment_dir,
        )
    else:
        raise NotImplementedError("Don't support multi-sequencial FASTA files")

    return feature_dict


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def main(args: argparse.Namespace):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    is_multimer = args.is_multimer

    if is_multimer:
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=args.kalign_binary_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=args.kalign_binary_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(monomer_data_pipeline=data_processor)

    output_dir_base = args.output_dir
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    tag_list = []
    seq_list = []
    with open(args.fasta_path, "r") as fp:
        fasta_str = fp.read()
    tags, seqs = parse_fasta(fasta_str)

    if not is_multimer and len(tags) != 1:
        logger.error(f"{args.fasta_path} contains more than one sequence but multimer mode is not enabled")

    tag_list.extend(tags)
    seq_list.extend(seqs)

    # Compute alignments
    precompute_alignments(tags, seqs, alignment_dir, args)

    # Genearte and dump feature dict
    feature_dict = generate_feature_dict(tags, seqs, alignment_dir, data_processor, args)

    pkl_output_path = os.path.join(output_dir_base, "features.pkz")
    logger.info(f"Write features on '{pkl_output_path}'")
    dump_pickle(feature_dict, pkl_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_path", type=str)
    parser.add_argument("template_mmcif_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--multimer", action="store_true", dest="is_multimer")
    parser.add_arugment("--non_pair", action="store_true", dest="non_pair")
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored.""",
    )
    add_data_args_(parser)

    args = parser.parse_args()

    main(args)
