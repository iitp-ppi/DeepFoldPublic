# Copyright 2023 DeepFold Team
# Copyright 2021 DeepMind Technologies Limited


"""Functions for building the features for AlphaFold-Multimer model."""


import contextlib
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np

import deepfold.model.alphafold.data.pipeline as pipeline_monomer
from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.data import parsers
from deepfold.data.tools import hhblits, hmmsearch, jackhmmer
from deepfold.model.alphafold.data import pipeline
from deepfold.model.alphafold.data.templates import HmmsearchHitFeaturizer
from deepfold.model.alphafold.data.types import FeatureDict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _FastaChain:
    sequence: str
    description: str


def _make_chain_id_map(
    *,
    sequences: Sequence[str],
    descriptions: Sequence[str],
) -> Mapping[str, _FastaChain]:
    """Makes a mapping from PDB-format chain ID to sequence and description."""

    # Check input consistency
    if len(sequences) != len(descriptions):
        raise ValueError(
            f"Sequences and descriptions must have equal lengths. Got {len(sequences)} != {len(descriptions)}"
        )
    # Too many chains
    if len(sequences) > protein.PDB_MAX_CHAINS:
        raise ValueError(f"Cannot process more chains than the PDB format supports. Got {len(sequences)} chains")

    chain_id_map = {}
    for chain_id, sequence, description in zip(protein.PDB_CHAIN_IDS, sequences, descriptions):
        chain_id_map[chain_id] = _FastaChain(sequence=sequence, description=description)

    return chain_id_map


@contextlib.contextmanager
def temp_fasta_file(fasta_str: str):
    with tempfile.NamedTemporaryFile("w", suffix=".fasta") as fasta_file:
        fasta_file.write(fasta_str)
        fasta_file.seek(0)
        yield fasta_file.name


def convert_monomer_features(
    monomer_features: pipeline.FeatureDict,
    chain_id: str,
) -> pipeline.FeatureDict:
    """Reshapes and modifies monomer features for multimer models."""
    converted = {}
    converted["auth_chain_id"] = np.asarray(chain_id, dtype=np.object_)

    unnecessary_leading_dim_feats = {"sequence", "domain_name", "num_alignments", "seq_length"}
    for feature_name, feature in monomer_features.items():
        if feature_name in unnecessary_leading_dim_feats:
            # Ensures it's a np.ndarray
            feature = np.asarray(feature[0], dtype=feature.dtype)
        elif feature_name == "aatype":
            # The multimer model performs the one-hot operation itself
            feature = np.argmax(feature, axis=-1).astype(np.int32)
        elif feature_name == "template_aatype":
            feature = np.argmax(feature, axis=-1).astype(np.int32)
            new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
            feature = np.take(new_order_list, feature, axis=0)
        elif feature_name == "template_all_atom_masks":
            feature_name == "template_all_atom_mask"
        converted[feature_name] = feature
    return converted


def int_id_to_str_id(num: int) -> str:
    """Encode a number as a string."""

    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}")

    num = num - 1  # 1-based indexing
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1

    return "".join(output)


def add_assembly_features(
    all_chain_features: MutableMapping[str, pipeline.FeatureDict],
) -> MutableMapping[str, pipeline.FeatureDict]:
    """
    Add features to distinguish between chains.

    Args:
        all_chain_features:
            A dictionary which maps chain_id to a dictionary of features for each chain.

    Returns:
        A dictionary which maps strings of the form `<seq_id>_<sym_id>` to the corresponding
        chain features.
    """
    # Group the chains by sequence
    seq_to_entity_id = {}
    grouped_chains = defaultdict(list)
    for _, chain_features in all_chain_features.items():
        seq = str(chain_features["sequence"])
        if seq not in seq_to_entity_id:
            seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
        grouped_chains[seq_to_entity_id[seq]].append(chain_features)

    new_all_chain_features = {}
    chain_id = 1
    for entity_id, group_chain_features in grouped_chains.items():
        for sym_id, chain_features in enumerate(group_chain_features, start=1):
            new_all_chain_features[f"{int_id_to_str_id(entity_id)}_{sym_id}"] = chain_features
            seq_length = chain_features["seq_length"]
            chain_features["asym_id"] = chain_id * np.ones(seq_length)
            chain_features["sym_id"] = sym_id * np.ones(seq_length)
            chain_features["entity_id"] = entity_id * np.ones(seq_length)
            chain_id += 1

    return new_all_chain_features


def pad_msa(
    np_example: pipeline.FeatureDict,
    min_num_seq: int,
) -> pipeline.FeatureDict:
    np_example = dict(np_example)
    num_seq = np_example["msa"].shape[0]
    if num_seq < min_num_seq:
        for feat in ("msas", "deletion_matrix", "bert_mask", "msa_mask"):
            np_example[feat] = np.pad(np_example[feat], ((0, min_num_seq - num_seq), (0, 0)))
        np_example["cluster_bias_mask"] = np.pad(np_example["cluster_bias_mask"], ((0, min_num_seq - num_seq),))
    return np_example


def run_msa_tool(
    msa_runner: Any,
    fasta_path: str,
    msa_out_path: str,
    msa_format: str,
    max_sto_sequences: Optional[int] = None,
) -> Mapping[str, Any]:
    """
    Runs an MSA tool.

    Notes:
        Check if output already exists first.
    """
    if msa_format == "sto" and max_sto_sequences is not None:
        result = msa_runner.query(fasta_path, max_sto_sequences)[0]
    else:
        result = msa_runner.query(fasta_path)[0]

    # Check format
    assert os.path.splitext(msa_out_path)[1] == msa_format

    with open(msa_out_path, "w") as fp:
        fp.write(result[msa_format])

    return result


class AlignmentRunnerMultimer:
    """Runs alignment tools and saves the results."""

    def __init__(
        self,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        uniref30_database_path: Optional[str] = None,
        uniprot_database_path: Optional[str] = None,
        template_searcher: Optional[hmmsearch.Hmmsearch] = None,
        use_small_bfd: Optional[bool] = None,
        num_cpus: Optional[int] = None,
        mgnify_max_hits: int = 5000,
        uniref_max_hits: int = 10000,
        uniprot_max_hits: int = 50000,
    ):
        """
        Args:
            jackhmmer_binary_path: str, Optional
                Path to jackhmmer binary
            hhblits_binary_path: str, Optional
                Path to hhblits binary
            uniref90_database_path: str, Optional
                Path to UniRef90 database.
                If provided, jackhmmer_binary_path must be provided.
            mgnify_database_path: str, Optional
                Path to MGnify database.
                If provided, jackhmmer_binary_path must be provided.
            bfd_database_path: str, Optional
                Path to BFD database.
                Depending on the value of use_small_bfd,
                one of hhblits_binary_path or jackhmmer_binary_path
                must be provided.
            uniref30_database_path:str, Optional
                Path to UniRef30.
                Search alongside BFD if use_small_bfd is false.
            uniprot_database_path: str, Optional
                Path to UniProt database.
                If provided, jackhmmer_binary_path must be provided.
            teamplate_searcher: HmmsearchHitFeaturizer, Optional
            use_small_bfd: str, Optional
                Whether to search the BFD database along with jackhmmer or
                in conjunction with UniRef30 with hhblits.
            num_cpus: int, Optional
                The number of CPUs available for alginment.
                By default, all CPUs are used.
        """
        db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                    uniprot_database_path,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
        }

        for name, dic in db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if binary is not None and not all([x is None for x in dbs]):
                raise ValueError(f"{name} DBs provided but {name} binary is None")

        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.uniprot_max_hits = uniprot_max_hits
        self.use_small_bfd = use_small_bfd

        if num_cpus is None:
            num_cpus = cpu_count()

        self.jackhmmer_uniref90_runner = None
        if jackhmmer_binary_path is not None and uniref90_database_path is not None:
            self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniref90_database_path,
                n_cpu=num_cpus,
            )

        self.jackhmmer_small_bfd_runner = None
        self.hhblits_bfd_uniref_runner = None
        if bfd_database_path is not None:
            if use_small_bfd:
                self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=bfd_database_path,
                    n_cpu=num_cpus,
                )
            else:
                dbs = [bfd_database_path]
                if uniref30_database_path is not None:
                    dbs.append(uniref30_database_path)
                self.hhblits_bfd_uniref_runner = hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=dbs,
                    n_cpu=num_cpus,
                )

        self.jackhmmer_mgnify_runner = None
        if mgnify_database_path is not None:
            self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=mgnify_database_path,
                n_cpu=num_cpus,
            )

        self.jackhmmer_uniprot_runner = None
        if uniprot_database_path is not None:
            self.jackhmmer_uniprot_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniprot_database_path,
                n_cpu=num_cpus,
            )

        if template_searcher is not None and self.jackhmmer_uniref90_runner is None:
            raise ValueError("Uniref90 runner must be specified to run template search")

        self.template_searcher = template_searcher

    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        """Runs alignment tools on the input sequence and create features."""

        with open(fasta_path, "r") as fp:
            fasta_str = fp.read()
        input_seqs, _ = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(f"More than one input sequence found in '{fasta_path}'")

        uniref90_out_path = os.path.join(output_dir, "uniref90_hits.sto")
        jackhmmer_uniref90_result = run_msa_tool(
            self.jackhmmer_uniref90_runner,
            fasta_path,
            uniref90_out_path,
            "sto",
            self.uniref_max_hits,
        )

        mgnify_out_path = os.path.join(output_dir, "mgnify_hits.sto")
        _ = run_msa_tool(
            self.jackhmmer_mgnify_runner,
            fasta_path,
            mgnify_out_path,
            "sto",
            self.mgnify_max_hits,
        )

        if self.template_searcher is not None:
            msa_for_templates = jackhmmer_uniref90_result["sto"]
            msa_for_templates = parsers.truncate_stockholm_msa(msa_for_templates, max_sequences=self.uniref_max_hits)
            msa_for_templates = parsers.deduplicate_stockholm_msa(msa_for_templates)
            msa_for_templates = parsers.remove_empty_columns_from_stockholm_msa(msa_for_templates)

            if self.template_searcher.input_format == "sto":
                pdb_templates_result = self.template_searcher.query(msa_for_templates)
            elif self.template_searcher.input_format == "a3m":
                uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(msa_for_templates)
                pdb_templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
            else:
                raise ValueError(f"Unrecognized template input format: {self.template_searcher.input_format}")

            pdb_hits_out_path = os.path.join(output_dir, f"pdb_hits.{self.template_searcher.output_format}")
            with open(pdb_hits_out_path, "w") as fp:
                fp.write(pdb_templates_result)

        if self.use_small_bfd:
            bfd_out_path = os.path.join(output_dir, "small_bfd_hits.sto")
            _ = run_msa_tool(
                self.jackhmmer_small_bfd_runner,
                fasta_path,
                bfd_out_path,
                "sto",
            )
        else:
            bfd_out_path = os.path.join(output_dir, "bfd_u")
            _ = run_msa_tool(
                self.hhblits_bfd_uniclust_runner,
                fasta_path,
                bfd_out_path,
                "a3m",
            )

        uniprot_path = os.path.join(output_dir, "uniprot_hits.sto")
        _ = run_msa_tool(
            self.jackhmmer_uniprot_runner,
            fasta_path,
            uniprot_path,
            "sto",
            self.uniprot_max_hits,
        )
