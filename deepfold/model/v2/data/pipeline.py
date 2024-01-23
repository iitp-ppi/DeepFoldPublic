# Copyright 2024 DeepFold Team


"""Functions for generating the input features for the model."""


import logging
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Union

import numpy as np

from deepfold.common import residue_constants as rc
from deepfold.model.v2.data import uniprot_identifiers
from deepfold.model.v2.data.ops import FeatureDict
from deepfold.search import parsers
from deepfold.search.templates import TemplateHit, TemplateHitFeaturizer, empty_template_feats
from deepfold.search.tools import hhblits, hhsearch, hmmsearch, jackhmmer

TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]
MsaRunner = Union[hhblits.HHBlits, hhsearch.HHSearch, hmmsearch.Hmmsearch, jackhmmer.Jackhmmer]

logger = logging.getLogger(__name__)


def make_sequence_features(
    sequence: str,
    description: str,
    num_res: int,
) -> FeatureDict:
    """
    Constructs a feature dictionary of sequence-related features.

    Args:
        sequence (str): The input sequence.
        description (str): A description or identifier for the sequence.
        num_res (int): The number of residues in the sequence.

    Returns:
        FeatureDict: A dictionary containing sequence-related features.

    """
    # Initialize an empty dictionary to store the sequence features
    features = {}

    # Encode the input sequence into one-hot representation
    features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )

    # Create an array of zeros representing between-segment residues
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)

    # Encode the description string into UTF-8 and store it as an object array
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)

    # Create an array of integers from 0 to 'num_res - 1' representing residue indices
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)

    # Create an array with all elements set to 'num_res' to represent sequence length
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)

    # Encode the input sequence into UTF-8 and store it as an object array
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)

    # Return the constructed feature dictionary
    return features


def make_msa_features(msas: Sequence[parsers.MSA]) -> FeatureDict:
    """
    Constructs a feature dictionary of MSA (Multiple Sequence Alignment) features.

    Args:
        msas (Sequence[parsers.MSA]): A sequence of MSA objects to process.

    Returns:
        FeatureDict: A dictionary containing MSA-related features.

    Raises:
        ValueError: If no MSA is provided or if an MSA doesn't contain at least one sequence.
    """
    # Check if at least one MSA is provided
    if not msas:
        raise ValueError("At least one MSA must be provided")

    int_msa = []  # List to store integer-encoded MSA sequences
    deletion_matrix = []  # List to store deletion matrices
    species_ids = []  # List to store species identifiers
    seen_sequences = set()  # Set to keep track of seen sequences

    # Iterate through the provided MSAs and their sequences to collect data
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence")
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)

            # Convert amino acids to integer IDs based on 'HHBLITS_AA_TO_ID' mapping
            try:
                int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in sequence])
            except Exception:
                print(msa_index, sequence_index)
                print(sequence)
                raise

            # Collect deletion matrices for each sequence in the MSA
            deletion_matrix.append(msa.deletion_matrix[sequence_index])

            # Extract and encode species identifiers in UTF-8 format
            identifiers = uniprot_identifiers.get_identifiers(msa.descriptions[sequence_index])
            species_ids.append(identifiers.species_id.encode("utf-8"))

    num_res = len(msas[0].sequences[0])  # Number of residues in the first sequence
    num_alignments = len(int_msa)  # Total number of MSA alignments

    # Initialize the feature dictionary
    features = {}

    # Store the deletion matrices as integer arrays
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)

    # Store the integer-encoded MSA sequences
    features["msa"] = np.array(int_msa, dtype=np.int32)

    # Store the number of alignments for each residue
    features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)

    # Store the species identifiers as object arrays
    features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)

    # Return the constructed feature dictionary
    return features


def make_dummy_msa_features(input_sequence: str) -> FeatureDict:
    """Generate 1-sequence MSA features having only the input sequence."""

    deletion_matrix = [[0 for _ in input_sequence]]
    msa_obj = parsers.MSA(sequences=[input_sequence], deletion_matrix=deletion_matrix, descriptions=[""])

    return make_msa_features([msa_obj])


def make_template_features(
    input_sequence: str,
    hits: Sequence[TemplateHit],
    template_featurizer: Optional[TemplateHitFeaturizer] = None,
) -> FeatureDict:
    """
    Constructs a feature dictionary of structural template-related features.

    Args:
        input_sequence (str): The input sequence for which features are to be generated.
        hits (Sequence[TemplateHit]): A sequence of template hits, each representing a structural template hit.
        template_featurizer (Optional[TemplateHitFeaturizer]): An optional template featurizer object used for feature extraction.

    Returns:
        FeatureDict: A dictionary containing structural template-related features for the input sequence.
    """

    # Combine all hits into a single list
    hits_cat = sum(hits.values(), [])

    # Check if there are no hits or if the template_featurizer is not provided
    if len(hits_cat) == 0 or template_featurizer is None:
        # Generate an empty template feature dictionary
        template_features = empty_template_feats(len(input_sequence))
    else:
        # Get template features using the provided featurizer
        template_result = template_featurizer.get_templates(query_sequence=input_sequence, hits=hits_cat)
        template_features = template_result.features

    # Return the computed template features
    return template_features


def run_msa_tool(
    msa_runner: MsaRunner, fasta_path: str, msa_out_path: str, msa_format: str, max_sto_sequences: Optional[int] = None
) -> Mapping[str, Any]:
    """
    Run an MSA (Multiple Sequence Alignment) tool, checking if the output already exists.

    Args:
        msa_runner (MsaRunner): An instance of the MSA runner responsible for running the MSA tool.
        fasta_path (str): The path to the input FASTA file containing sequences to be aligned.
        msa_out_path (str): The path to save the MSA output.
        msa_format (str): The desired format of the MSA output (e.g., "sto" for Stockholm format).
        max_sto_sequences (Optional[int]): The maximum number of sequences to include in the output when the format is "sto".

    Returns:
        Mapping[str, Any]: A dictionary containing the MSA result. The dictionary typically contains the
            MSA output data under the key specified by `msa_format`.

    Raises:
        AssertionError: If the file extension of `msa_out_path` does not match the specified `msa_format`.
    """

    # Query the MSA runner to obtain the MSA result
    if msa_format == "sto" and max_sto_sequences is not None:
        result = msa_runner.query(fasta_path, max_sto_sequences)[0]
    elif msa_format == "a3m":
        result = msa_runner.query(fasta_path)
    else:
        ValueError(f"Unrecognized MSA format '{msa_format}'")

    # Check if the file extension of msa_out_path matches the specified msa_format
    assert os.path.splitext(msa_out_path)[-1][1:] == msa_format

    # Write the MSA result to the specified output file
    with open(msa_out_path, "w") as fp:
        fp.write(result[msa_format])

    # Return the MSA result
    return result


class AlignmentRunner:
    """Runs alignment tools and saves the resutls."""

    def __init__(
        self,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        uniref30_database_path: Optional[str] = None,
        uniprot_database_path: Optional[str] = None,
        template_searcher: Optional[TemplateSearcher] = None,
        use_small_bfd: bool = False,
        num_cpus: Optional[int] = None,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
        uniprot_max_hits: int = 50000,
    ):
        """
        Initializes an AlignmentRunner instance.

        Args:
            jackhmmer_binary_path (str, optional): Path to the Jackhmmer binary.
            hhblits_binary_path (str, optional): Path to the HHBlits binary.
            uniref90_database_path (str, optional): Path to the UniRef90 database.
            mgnify_database_path (str, optional): Path to the MGnify database.
            bfd_database_path (str, optional): Path to the BFD database.
            uniref30_database_path (str, optional): Path to the UniRef30 database.
            uniprot_database_path (str, optional): Path to the UniProt database.
            template_searcher (TemplateSearcher, optional): Template searcher instance.
            use_small_bfd (bool, optional): Whether to use the small BFD database.
            num_cpus (int, optional): Number of CPU cores to use for parallel processing.
            uniref_max_hits (int): Maximum number of UniRef90 hits to save.
            mgnify_max_hits (int): Maximum number of MGnify hits to save.
            uniprot_max_hits (int): Maximum number of UniProt hits to save.
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
                "dbs": [bfd_database_path if not use_small_bfd else None],
            },
        }

        for name, dic in db_map.items():
            binary, dbs = dict["binary"], dic["dbs"]
            if binary is None and not all([x is None for x in dbs]):
                raise ValueError(f"Databases are provided but {name} binary is None")

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
            raise ValueError("UniRef90 runner must be specified to run template search")

        self.template_searcher = template_searcher

    def run(self, fasta_path: str, output_dir: str):
        """
        Runs alignment tools on a sequence and saves the results in the specified output directory.

        Args:
            fasta_path (str): Path to the input FASTA file.
            output_dir (str): Path to the directory where the results will be saved.
        """

        if self.jackhmmer_uniref90_runner is not None:
            uniref90_out_path = os.path.join(output_dir, "uniref90_hits.sto")

            jackhmmer_uniref90_result = run_msa_tool(
                msa_runner=self.jackhmmer_uniref90_runner,
                fasta_path=fasta_path,
                msa_out_path=uniref90_out_path,
                msa_format="sto",
                max_sto_sequences=self.uniref_max_hits,
            )

            template_msa = jackhmmer_uniref90_result["sto"]
            template_msa = parsers.deduplicate_stockholm_msa(template_msa)
            template_msa = parsers.remove_empty_columns_from_stockholm_msa(template_msa)

            if self.template_searcher is not None:
                if self.template_searcher.input_format == "sto":
                    templates_result = self.template_searcher.query(template_msa)
                elif self.template_searcher.input_format == "a3m":
                    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(template_msa)
                    templates_result = self.template_searcher.query(uniref90_msa_as_a3m)
                else:
                    output_fmt = self.template_searcher.input_format
                    raise ValueError(f"Unrecognized template input format '{output_fmt}'")

            output_fmt = self.template_searcher.output_format
            if output_fmt == "sto":
                template_hits_name = "hmm_output"
            elif output_fmt == "hhr":
                template_hits_name = "hhr_output"

            template_hits_out_path = os.path.join(output_dir, f"{template_hits_name}.{output_fmt}")
            with open(template_hits_out_path, "w") as fp:
                fp.write(templates_result)

        if self.jackhmmer_mgnify_runner is not None:
            mgnify_out_path = os.path.join(output_dir, "mgnify_hits.sto")
            _ = run_msa_tool(
                msa_runner=self.jackhmmer_mgnify_runner,
                fasta_path=fasta_path,
                msa_out_path=mgnify_out_path,
                msa_format="sto",
                max_sto_sequences=self.mgnify_max_hits,
            )

        if self.use_small_bfd and self.jackhmmer_small_bfd_runner is not None:
            bfd_out_path = os.path.join(output_dir, "small_bfd_hits.sto")
            _ = run_msa_tool(
                msa_runner=self.jackhmmer_small_bfd_runner,
                fasta_path=fasta_path,
                msa_out_path=bfd_out_path,
                msa_format="sto",
            )
        elif self.hhblits_bfd_uniref_runner is not None:
            bfd_out_path = os.path.join(output_dir, "bfd_uniref30_hits.a3m")
            _ = run_msa_tool(
                msa_runner=self.hhblits_bfd_uniref_runner,
                fasta_path=fasta_path,
                msa_out_path=bfd_out_path,
                msa_format="a3m",
            )

        if self.jackhmmer_uniprot_runner is not None:
            uniprot_out_path = os.path.join(output_dir, "uniprot_hits.sto")
            _ = run_msa_tool(
                self.jackhmmer_uniprot_runner,
                fasta_path=fasta_path,
                msa_out_path=uniprot_out_path,
                msa_format="sto",
                max_sto_sequences=self.uniprot_max_hits,
            )


class DataPipeline:
    """Assemble input features."""

    def __init__(
        self,
        template_featurizer: Optional[TemplateHitFeaturizer],
        alphafold_mode: bool = True,
    ):
        self.template_featurizer = template_featurizer
        self.alphafold_mode = alphafold_mode

    def _parse_template_hit_files(
        self,
        alignment_dir: str,
        input_sequence: str,
    ) -> Mapping[str, Sequence[TemplateHit]]:
        all_hits = {}

        if self.alphafold_mode:
            path = Path(alignment_dir)

            pdb_hits_path = path / "pdb_hits.hhr"
            hmm_output_path = path / "hmm_output.sto"

            if pdb_hits_path.exists():
                with open(pdb_hits_path, "r") as fp:
                    hits = parsers.parse_hhr(fp.read())
                    all_hits["pdb_hits.hrr"] = hits
            elif hmm_output_path.exists():
                with open(hmm_output_path, "r") as fp:
                    hits = parsers.parse_hmmsearch_sto(fp.read(), input_sequence)
                    all_hits["hmm_output.sto"] = hits
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if ext == ".hhr":
                    with open(path, "r") as fp:
                        hits = parsers.parse_hhr(fp.read())
                    all_hits[f] = hits
                elif f == "hmm_output.sto":
                    with open(path, "r") as fp:
                        hits = parsers.parse_hmmsearch_sto(fp.read(), input_sequence)
                    all_hits[f] = hits

        return all_hits

    def _process_msa_feats(
        self,
        alignment_dir: str,
        input_sequence: Optional[str] = None,
    ) -> FeatureDict:
        msas = self._get_msas(alignment_dir, input_sequence)
        msa_features = make_msa_features(msas)

        return msa_features

    def _get_msas(self, alignment_dir: str, input_sequence: Optional[str] = None) -> List[parsers.MSA]:
        msa_data = self._parse_msa_data(alignment_dir)

        if len(msa_data) == 0:
            if input_sequence is None:
                raise ValueError("If the alignment directory contains no MSAs, an input sequence must be provided")
            msa_data["dummy"] = make_dummy_msa_features(input_sequence)

        return list(msa_data.values())

    def _parse_msa_data(self, alignment_dir: str) -> Mapping[str, parsers.MSA]:
        msa_data = {}

        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            filename, ext = os.path.splitext(f)
            filename = os.path.split(filename)[-1]

            if ext == ".a3m":
                with open(path, "r") as fp:
                    msas = parsers.parse_a3m(fp.read())
                data = {
                    "sequences": msas.sequences,
                    "deletion_matrix": msas.deletion_matrix,
                    "descriptions": msas.descriptions,
                }
            elif ext == ".sto":
                if self.alphafold_mode and filename in ("uniprot_hits", "hmm_output"):
                    continue
                with open(path, "r") as fp:
                    msas = parsers.parse_stockholm(fp.read())
                data = {
                    "sequences": msas.sequences,
                    "deletion_matrix": msas.deletion_matrix,
                    "descriptions": msas.descriptions,
                }
            else:
                continue

            msa_data[f] = parsers.MSA(**data)

        return msa_data

    def process_fasta(
        self,
        fasta_path: str,
        alignment_dir: str,
    ) -> FeatureDict:
        """Assemble features for a single sequence in a FASTA file."""
        with open(fasta_path) as fp:
            fasta_str = fp.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(f"More than one input sequence found in '{fasta_path}'")
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        hits = self._parse_template_hit_files(alignment_dir, input_sequence)
        template_features = make_template_features(input_sequence, hits, template_featurizer=self.template_featurizer)

        sequence_features = make_sequence_features(input_sequence, input_description, num_res)

        msa_features = self._process_msa_feats(alignment_dir, input_sequence=input_sequence)

        return {
            **sequence_features,
            **msa_features,
            **template_features,
        }
