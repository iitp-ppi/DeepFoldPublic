# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import logging
import os
from typing import Any, Mapping, Optional

import torch
from omegaconf import DictConfig

from deepfold.common import protein
from deepfold.data import mmcif_parsing, parsers, templates
from deepfold.model.alphafold.data.make_feats import (
    _aatype_to_str_sequence,
    make_dummy_msa_feats,
    make_mmcif_features,
    make_msa_features,
    make_pdb_features,
    make_protein_features,
    make_sequence_features,
    make_template_features,
    unify_template_features,
)
from deepfold.model.alphafold.data.proc_feats import np_example_to_features
from deepfold.model.alphafold.data.types import FeatureDict, TensorDict

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self, cfg: DictConfig) -> None:
        self.config = cfg

    def process(self, raw_features: FeatureDict, mode: str = "predict") -> TensorDict:
        return np_example_to_features(np_example=raw_features, cfg=self.config, mode=mode)


class DataPipeline:
    """Assembles input features."""

    def __init__(
        self,
        template_featurizer: Optional[templates.TemplateHitFeaturizer],
    ):
        self.template_featurizer = template_featurizer

    def _parse_msa_data(
        self,
        alignment_dir: str,
        alignment_index: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        msa_data = {}
        if alignment_index is not None:
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), "rb")

            def read_msa(start, size):
                fp.seek(start)
                msa = fp.read(size).decode("utf-8")
                return msa

            for name, start, size in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if ext == ".a3m":
                    msa, deletion_matrix = parsers.parse_a3m(read_msa(start, size))
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                elif ext == ".sto":
                    msa, deletion_matrix, _ = parsers.parse_stockholm(read_msa(start, size))
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                else:
                    continue

                msa_data[name] = data

            fp.close()
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if ext == ".a3m":
                    with open(path, "r") as fp:
                        msa, deletion_matrix = parsers.parse_a3m(fp.read())
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                elif ext == ".sto":
                    with open(path, "r") as fp:
                        msa, deletion_matrix, _ = parsers.parse_stockholm(fp.read())
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                else:
                    continue

                msa_data[f] = data

        return msa_data

    def _parse_template_hits(self, alignment_dir: str, alignment_index: Optional[Any] = None) -> Mapping[str, Any]:
        all_hits = {}
        if alignment_index is not None:
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), "rb")

            def read_template(start, size):
                fp.seek(start)
                return fp.read(size).decode("utf-8")

            for name, start, size in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if ext == ".hhr":
                    hits = parsers.parse_hhr(read_template(start, size))
                    all_hits[name] = hits

            fp.close()
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if ext == ".hhr":
                    with open(path, "r") as fp:
                        hits = parsers.parse_hhr(fp.read())
                    all_hits[f] = hits

        return all_hits

    def _get_msas(
        self,
        alignment_dir: str,
        input_sequence: Optional[str] = None,
        alignment_index: Optional[str] = None,
    ):
        msa_data = self._parse_msa_data(alignment_dir, alignment_index)
        if len(msa_data) == 0:
            if input_sequence is None:
                raise ValueError(
                    """
                    If the alignment dir contains no MSAs, an input sequence 
                    must be provided.
                    """
                )
            msa_data["dummy"] = {
                "msa": [input_sequence],
                "deletion_matrix": [[0 for _ in input_sequence]],
            }

        msas, deletion_matrices = zip(*[(v["msa"], v["deletion_matrix"]) for v in msa_data.values()])

        return msas, deletion_matrices

    def _process_msa_feats(
        self, alignment_dir: str, input_sequence: Optional[str] = None, alignment_index: Optional[str] = None
    ) -> Mapping[str, Any]:
        msas, deletion_matrices = self._get_msas(alignment_dir, input_sequence, alignment_index)
        msa_features = make_msa_features(
            msas=msas,
            deletion_matrices=deletion_matrices,
        )

        return msa_features

    # Load and process sequence embedding features
    def _process_seqemb_features(
        self,
        alignment_dir: str,
    ) -> Mapping[str, Any]:
        seqemb_features = {}
        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            ext = os.path.splitext(f)[-1]

            if ext == ".pt":
                # Load embedding file
                seqemb_data = torch.load(path)
                seqemb_features["seq_embedding"] = seqemb_data["representations"][33]

        return seqemb_features

    def process_fasta(
        self,
        fasta_path: str,
        alignment_dir: str,
        alignment_index: Optional[str] = None,
        seqemb_mode: bool = False,
    ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file"""
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(f"More than one input sequence found in {fasta_path}.")
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        sequence_embedding_features = {}
        # If using seqemb mode, generate a dummy MSA features using just the sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
            sequence_embedding_features = self._process_seqemb_features(alignment_dir)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {
            **sequence_features,
            **msa_features,
            **template_features,
            **sequence_embedding_features,
        }

    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
        alignment_dir: str,
        chain_id: Optional[str] = None,
        alignment_index: Optional[str] = None,
        seqemb_mode: bool = False,
    ) -> FeatureDict:
        """
        Assembles features for a specific chain in an mmCIF object.

        If chain_id is None, it is assumed that there is only one chain
        in the object. Otherwise, a ValueError is thrown.
        """
        if chain_id is None:
            chains = mmcif.structure.get_chains()
            chain = next(chains, None)
            if chain is None:
                raise ValueError("No chains in mmCIF file")
            chain_id = chain.id

        mmcif_feats = make_mmcif_features(mmcif, chain_id)

        input_sequence = mmcif.chain_to_seqres[chain_id]
        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
            query_release_date=templates.to_date(mmcif.header["release_date"]),
        )

        sequence_embedding_features = {}
        # If using seqemb mode, generate a dummy MSA features using just the sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
            sequence_embedding_features = self._process_seqemb_features(alignment_dir)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {**mmcif_feats, **template_features, **msa_features, **sequence_embedding_features}

    def process_pdb(
        self,
        pdb_path: str,
        alignment_dir: str,
        is_distillation: bool = True,
        chain_id: Optional[str] = None,
        _structure_index: Optional[str] = None,
        alignment_index: Optional[str] = None,
        seqemb_mode: bool = False,
    ) -> FeatureDict:
        """
        Assembles features for a protein in a PDB file.
        """
        if _structure_index is not None:
            db_dir = os.path.dirname(pdb_path)
            db = _structure_index["db"]
            db_path = os.path.join(db_dir, db)
            fp = open(db_path, "rb")
            _, offset, length = _structure_index["files"][0]
            fp.seek(offset)
            pdb_str = fp.read(length).decode("utf-8")
            fp.close()
        else:
            with open(pdb_path, "r") as f:
                pdb_str = f.read()

        protein_object = protein.from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(protein_object, description, is_distillation=is_distillation)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_embedding_features = {}
        # If in sequence embedding mode, generate dummy MSA features using just the input sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
            sequence_embedding_features = self._process_seqemb_features(alignment_dir)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {**pdb_feats, **template_features, **msa_features, **sequence_embedding_features}

    def process_core(
        self,
        core_path: str,
        alignment_dir: str,
        alignment_index: Optional[str] = None,
        seqemb_mode: bool = False,
    ) -> FeatureDict:
        """
        Assembles features for a protein in a ProteinNet .core file.
        """
        with open(core_path, "r") as f:
            core_str = f.read()

        protein_object = protein.from_proteinnet_string(core_str)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(core_path))[0].upper()
        core_feats = make_protein_features(protein_object, description)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_embedding_features = {}
        # If in sequence embedding mode, generate dummy MSA features using just the input sequence
        if seqemb_mode:
            msa_features = make_dummy_msa_feats(input_sequence)
            sequence_embedding_features = self._process_seqemb_features(alignment_dir)
        else:
            msa_features = self._process_msa_feats(alignment_dir, input_sequence)

        return {**core_feats, **template_features, **msa_features, **sequence_embedding_features}

    def process_multiseq_fasta(
        self,
        fasta_path: str,
        super_alignment_dir: str,
        ri_gap: int = 200,
    ) -> FeatureDict:
        """
        Assembles features for a multi-sequence FASTA. Uses Minkyung Baek's hack from Twitter (a.k.a. AlphaFold-Gap).
        """
        with open(fasta_path, "r") as f:
            fasta_str = f.read()

        input_seqs, input_descs = parsers.parse_fasta(fasta_str)

        # No whitespace allowed
        input_descs = [i.split()[0] for i in input_descs]

        # Stitch all of the sequences together
        input_sequence = "".join(input_seqs)
        input_description = "-".join(input_descs)
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        seq_lens = [len(s) for s in input_seqs]
        total_offset = 0
        for sl in seq_lens:
            total_offset += sl
            sequence_features["residue_index"][total_offset:] += ri_gap

        msa_list = []
        deletion_mat_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(super_alignment_dir, desc)
            msas, deletion_mats = self._get_msas(alignment_dir, seq, None)
            msa_list.append(msas)
            deletion_mat_list.append(deletion_mats)

        final_msa = []
        final_deletion_mat = []
        msa_it = enumerate(zip(msa_list, deletion_mat_list))
        for i, (msas, deletion_mats) in msa_it:
            prec, post = sum(seq_lens[:i]), sum(seq_lens[i + 1 :])
            msas = [[prec * "-" + seq + post * "-" for seq in msa] for msa in msas]
            deletion_mats = [[prec * [0] + dml + post * [0] for dml in deletion_mat] for deletion_mat in deletion_mats]

            assert len(msas[0][-1]) == len(input_sequence)

            final_msa.extend(msas)
            final_deletion_mat.extend(deletion_mats)

        msa_features = make_msa_features(
            msas=final_msa,
            deletion_matrices=final_deletion_mat,
        )

        template_feature_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(super_alignment_dir, desc)
            hits = self._parse_template_hits(alignment_dir, alignment_index=None)
            template_features = make_template_features(
                seq,
                hits,
                self.template_featurizer,
            )
            template_feature_list.append(template_features)

        template_features = unify_template_features(template_feature_list)

        return {
            **sequence_features,
            **msa_features,
            **template_features,
        }
