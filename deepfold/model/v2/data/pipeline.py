import copy
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

import deepfold.model.v2.data.pairing.uniprot_identifiers as msa_identifiers
from deepfold.common import protein
from deepfold.common import residue_constants as rc
from deepfold.model.v2.data.merge import pair_and_merge
from deepfold.model.v2.data.pairing import baseline as msa_pairing
from deepfold.model.v2.data.utils import add_assembly_features, convert_monomer_features, pad_msa, temp_fasta_file
from deepfold.model.v2.search.utils import SchemeRegularizer
from deepfold.search import mmcif_parsing, parsers, templates
from deepfold.search.templates import empty_template_feats
from deepfold.search.tools import hhsearch, hmmsearch

FeatureDict = Dict[str, np.ndarray]
TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


def make_template_features(
    input_sequence: str,
    hits: Dict[str, Sequence[templates.TemplateHit]],
    template_featurizer: templates.TemplateHitFeaturizer,
) -> FeatureDict:
    """Construct a feature dict of template features."""

    hits_cat = sum(hits.values(), [])
    if len(hits_cat) == 0 or template_featurizer is None:
        template_features = empty_template_feats(len(input_sequence))
    else:
        template_result = template_featurizer.get_templates(input_sequence, hits_cat)
        template_features = template_result.features

    return dict(template_features)


def unify_template_features(template_feature_list: Sequence[FeatureDict]) -> FeatureDict:
    out_dicts = []
    seq_lens = [fd["template_aatype"].shape[1] for fd in template_feature_list]
    for i, fd in enumerate(template_feature_list):
        out_dict = {}
        n_templates, n_res = fd["template_aatype"].shape[:2]
        for k, v in fd.items():
            seq_keys = [
                "template_aatype",
                "template_all_atom_positions",
                "template_all_atom_mask",
            ]
            if k in seq_keys:
                new_shape = list(v.shape)
                assert new_shape[1] == n_res
                new_shape[1] = sum(seq_lens)
                new_array = np.zeros(new_shape, dtype=v.dtype)

                if k == "template_aatype":
                    new_array[..., rc.HHBLITS_AA_TO_ID["-"]] = 1

                offset = sum(seq_lens[:i])
                new_array[:, offset : offset + seq_lens[i]] = v
                out_dict[k] = new_array
            else:
                out_dict[k] = v

        chain_indices = np.array(n_templates * [i])
        out_dict["template_chain_index"] = chain_indices

        if n_templates != 0:
            out_dicts.append(out_dict)

    if len(out_dicts) > 0:
        out_dict = {k: np.concatenate([od[k] for od in out_dicts]) for k in out_dicts[0]}
    else:
        out_dict = empty_template_feats(sum(seq_lens))

    return out_dict


def make_sequence_features(sequence: str, description: str, num_res: int) -> FeatureDict:
    """Construct a feature dict of sequence features."""

    features = {}
    features["aatype"] = rc.sequence_to_onehot(
        sequence=sequence,
        mapping=rc.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=object)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=object)

    return features


def make_mmcif_features(mmcif_object: mmcif_parsing.MmcifObject, chain_id: str) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(make_sequence_features(input_sequence, description, num_res))

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(mmcif_object, chain_id)
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array([mmcif_object.header["resolution"]], dtype=np.float32)
    mmcif_feats["release_date"] = np.array([mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_)

    mmcif_feats["is_distillation"] = np.array(0.0, dtype=np.float32)

    return mmcif_feats


def aatype_to_str_sequence(aatype):
    return "".join([rc.restypes_with_x[aatype[i]] for i in range(len(aatype))])


def make_protein_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.0]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.0 if is_distillation else 0.0).astype(np.float32)

    return pdb_feats


def make_msa_features(msas: Sequence[parsers.Msa]) -> FeatureDict:
    """Constructs a feature dict of MSA features."""

    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(f"MSA {msa_index} must contain at least one sequence.")
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in sequence])

            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            identifiers = msa_identifiers.get_identifiers(msa.descriptions[sequence_index])
            species_ids.append(identifiers.species_id.encode("utf-8"))

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array([num_alignments] * num_res, dtype=np.int32)
    features["msa_species_identifiers"] = np.array(species_ids, dtype=np.object_)
    return features


def make_dummy_msa_object(input_sequence) -> parsers.Msa:
    deletion_matrix = [[0 for _ in input_sequence]]
    return parsers.Msa(sequences=[input_sequence], deletion_matrix=deletion_matrix, descriptions=["dummy"])


def make_dummy_msa_feats(input_sequence) -> FeatureDict:
    msa_data_obj = make_dummy_msa_object(input_sequence)
    return make_msa_features([msa_data_obj])


class DataPipeline:
    """Assembles input features."""

    def __init__(
        self,
        template_featurizer: Optional[templates.TemplateHitFeaturizer],
    ) -> None:
        self.template_featurizer = template_featurizer

    def parse_template_hit_files(
        self,
        alignment_dir: str,
        input_sequence: str,
        alignment_scheme: Dict[str, List[str]],
    ) -> Dict[str, Sequence[templates.TemplateHit]]:
        all_hits = {}

        for name in alignment_scheme["template"]:
            path = os.path.join(alignment_dir, name)
            ext = os.path.splitext(name)[-1]

            if ext == ".hhr":
                with open(path, "r") as fp:
                    hits = parsers.parse_hhr(fp.read())
                all_hits[name] = hits
            elif ext == ".sto":
                with open(path, "r") as fp:
                    hits = parsers.parse_hmmsearch_sto(fp.read(), input_sequence)
                all_hits[name] = hits
            elif ext == ".a3m":
                with open(path, "r") as fp:
                    hits = parsers.parse_hmmsearch_a3m(input_sequence, fp.read())
                all_hits[name] = hits
            else:
                raise ValueError(f"Unexpected extension: '{ext[1:]}'")

        return all_hits

    def parse_msa_data(
        self,
        alignment_dir: str,
        alignment_scheme: Dict[str, List[str]],
    ) -> Dict[str, parsers.MSA]:
        msa_data = {}

        for name in alignment_scheme["msa"]:
            path = os.path.join(alignment_dir, name)
            ext = os.path.splitext(name)[-1]

            if ext == ".sto":
                with open(path, "r") as fp:
                    msa = parsers.parse_stockholm(fp.read())
                msa_data[name] = msa
            elif ext == ".a3m":
                with open(path, "r") as fp:
                    msa = parsers.parse_a3m(fp.read())
                msa_data[name] = msa
            else:
                raise ValueError(f"Unexpected extension: '{ext[1:]}'")

        return msa_data

    def process_msa_feats(
        self,
        alignment_dir: str,
        input_sequence: str,
        alignment_scheme: Dict[str, List[str]],
    ) -> FeatureDict:
        msa_data = self.parse_msa_data(alignment_dir, alignment_scheme)
        if len(msa_data) == 0:
            msa_data["dummy"] = make_dummy_msa_object(input_sequence)
        msas = list(msa_data.values())
        msa_features = make_msa_features(msas)

        return msa_features

    def process_fasta(
        self,
        fasta_path: str,
        alignment_dir: str,
        alignment_scheme: Dict[str, List[str]],
    ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file."""

        with open(fasta_path) as fp:
            fasta_str = fp.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(f"More than one input sequence found in '{fasta_path}'")
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        sequence_featuers = make_sequence_features(input_sequence, input_description, num_res)

        all_hits = self.parse_template_hit_files(alignment_dir, input_sequence, alignment_scheme)
        template_features = make_template_features(input_sequence, all_hits, self.template_featurizer)

        msa_features = self.process_msa_feats(alignment_dir, input_sequence, alignment_scheme)

        return {**sequence_featuers, **msa_features, **template_features}

    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
        alignment_dir: str,
        chain_id: str,
        alignment_scheme: Dict[str, List[str]],
    ) -> FeatureDict:
        """Assembles features for a specific chain in an mmCIF object."""

        mmcif_feats = make_mmcif_features(mmcif, chain_id)

        input_sequence = mmcif.chain_to_seqres[chain_id]

        hits = self.parse_template_hit_files(alignment_dir, input_sequence, alignment_scheme)
        template_features = make_template_features(input_sequence, hits, self.template_featurizer)

        msa_features = self.process_msa_feats(alignment_dir, input_sequence, alignment_scheme)

        return {**mmcif_feats, **template_features, **msa_features}

    def process_multiseq_fasta(
        self,
        fasta_path: str,
        super_alignment_dir: str,
        ri_gap: int = 200,
    ) -> FeatureDict:
        """Assembles features for a multi-sequence FASTA. AKA AF-Gap."""

        alignment_scheme = {
            "template": ["template_hits.hhr"],
            "msa": ["*.sto", "*.a3m"],
        }

        with open(fasta_path, "r") as f:
            fasta_str = f.read()

        input_seqs, input_descs = parsers.parse_fasta(fasta_str)

        # No whitespace allowed
        input_descs = [i.split()[0] for i in input_descs]

        # Stitch all of the sequences together
        input_sequence = "".join(input_seqs)
        input_description = "-".join(input_descs)
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(input_sequence, input_description, num_res)

        seq_lens = [len(s) for s in input_seqs]
        total_offset = 0
        for sl in seq_lens:
            total_offset += sl
            sequence_features["residue_index"][total_offset:] += ri_gap

        msa_list = []
        deletion_mat_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(super_alignment_dir, desc)
            msas = self._get_msas(alignment_dir, seq, alignment_scheme)
            msa_list.append([m.sequences for m in msas])
            deletion_mat_list.append([m.deletion_matrix for m in msas])

        final_msa = []
        final_deletion_mat = []
        final_msa_obj = []
        msa_it = enumerate(zip(msa_list, deletion_mat_list))
        for i, (msas, deletion_mats) in msa_it:
            prec, post = sum(seq_lens[:i]), sum(seq_lens[i + 1 :])
            msas = [[prec * "-" + seq + post * "-" for seq in msa] for msa in msas]
            deletion_mats = [[prec * [0] + dml + post * [0] for dml in deletion_mat] for deletion_mat in deletion_mats]

            assert len(msas[0][-1]) == len(input_sequence)

            final_msa.extend(msas)
            final_deletion_mat.extend(deletion_mats)
            final_msa_obj.extend(
                [
                    parsers.Msa(sequences=msas[k], deletion_matrix=deletion_mats[k], descriptions=None)
                    for k in range(len(msas))
                ]
            )

        msa_features = make_msa_features(msas=final_msa_obj)

        template_feature_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(super_alignment_dir, desc)
            hits = self.parse_template_hit_files(alignment_dir, seq, alignment_scheme)

            template_features = make_template_features(seq, hits, self.template_featurizer)
            template_feature_list.append(template_features)

        template_features = unify_template_features(template_feature_list)

        return {**sequence_features, **msa_features, **template_features}


class DataPipelineMultimer:
    """Assembles the input features."""

    def __init__(
        self,
        monomer_data_pipeline: DataPipeline,
    ) -> None:
        self.monomer_data_pipeline = monomer_data_pipeline
        self.alignment_scheme = {
            "template": ["template_hits.sto"],
            "uniprot": ["uniprot_hits.sto"],
            "msa": ["*.sto", "*.a3m"],
        }

    def process_single_chain(
        self,
        chain_id: str,
        sequence: str,
        chain_alignment_dir: str,
        is_homomer_or_monomer: bool,
    ) -> FeatureDict:
        """Runs the monomer pipeline on a single chain."""

        chain_fasta_str = f">{chain_id}\n{sequence}\n"

        if not os.path.exists(chain_alignment_dir):
            raise ValueError(f"Alignment for {chain_id} not found")

        with temp_fasta_file(chain_fasta_str) as chain_fasta_path:
            alignment_scheme = SchemeRegularizer(
                key_order=["template", "uniprot", "msa"],
                base_dir=chain_alignment_dir,
            ).process(self.alignment_scheme)
            chain_features = self.monomer_data_pipeline.process_fasta(
                fasta_path=chain_fasta_path,
                alignment_dir=chain_alignment_dir,
                alignment_scheme=alignment_scheme,
            )

            # Only construct the paring features if there are two or more unique sequences.
            if not is_homomer_or_monomer:
                all_seq_msa_features = self.all_seq_msa_features(chain_alignment_dir)
                chain_features.update(all_seq_msa_features)

        return chain_features

    @staticmethod
    def all_seq_msa_features(alignment_dir: str) -> FeatureDict:
        """Get MSA features for unclustered UniProt for pairing."""

        uniprot_msa_path = os.path.join(alignment_dir, "uniprot_hits.sto")
        if not os.path.exists(uniprot_msa_path):
            chain_id = os.path.basename(os.path.normpath(alignment_dir))
            raise ValueError(f"Missing 'uniprot_hits.sto' for {chain_id} which is required for MSA pairing")

        with open(uniprot_msa_path, "r") as fp:
            uniprot_msa_string = fp.read()
        msa = parsers.parse_stockholm(uniprot_msa_string)

        all_seq_features = make_msa_features([msa])
        valid_feats = msa_pairing.MSA_FEATURES + ("msa_species_identifiers")
        feats = {f"{k}_all_seq": v for k, v in all_seq_features.items() if k in valid_feats}

        return feats

    def process_fasta(self, fasta_path: str, alignment_dir: str) -> FeatureDict:
        with open(fasta_path, "r") as fp:
            input_fasta_str = fp.read()

        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

        all_chain_features = {}
        sequence_features = {}
        is_homomer_or_monomer = len(set(input_seqs)) == 1
        for seq, desc in zip(input_seqs, input_descs):
            if seq in sequence_features:
                all_chain_features[desc] = copy.deepcopy(sequence_features[seq])
                continue

            chain_alignment_dir = os.path.join(alignment_dir, desc)
            chain_features = self.process_single_chain(
                chain_id=desc,
                sequence=seq,
                chain_alignment_dir=chain_alignment_dir,
                is_homomer_or_monomer=is_homomer_or_monomer,
            )
            chain_features = convert_monomer_features(chain_features)

            all_chain_features[desc] = chain_features
            sequence_features[seq] = chain_features

        all_chain_features = add_assembly_features(all_chain_features)

        # Pair and merge chain features.
        np_example = pair_and_merge(all_chain_features)

        # Pad MSA to avoid zero-sized extra MSA.
        np_example = pad_msa(np_example, 512)

        return np_example

    def get_mmcif_features(
        self,
        mmcif_object: mmcif_parsing.MmcifObject,
        chain_id: str,
    ) -> FeatureDict:
        mmcif_feats = {}

        all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(mmcif_object=mmcif_object, chain_id=chain_id)
        mmcif_feats["all_atom_positions"] = all_atom_positions
        mmcif_feats["all_atom_mask"] = all_atom_mask

        mmcif_feats["resolution"] = np.array(mmcif_object.header["resolution"], dtype=np.float32)

        mmcif_feats["release_date"] = np.array([mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_)

        mmcif_feats["is_distillation"] = np.array(0.0, dtype=np.float32)

        return mmcif_feats

    def process_mmcif(
        self,
        mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
        alignment_dir: str,
    ) -> FeatureDict:

        all_chain_features = {}
        sequence_features = {}
        is_homomer_or_monomer = len(set(list(mmcif.chain_to_seqres.values()))) == 1
        for chain_id, seq in mmcif.chain_to_seqres.items():
            desc = "_".join([mmcif.file_id, chain_id])

            if seq in sequence_features:
                all_chain_features[desc] = copy.deepcopy(sequence_features[seq])
                continue

            chain_alignment_dir = os.path.join(alignment_dir, desc)
            chain_features = self.process_single_chain(
                chain_id=desc,
                sequence=seq,
                chain_alignment_dir=chain_alignment_dir,
                is_homomer_or_monomer=is_homomer_or_monomer,
            )

            chain_features = convert_monomer_features(chain_features)

            mmcif_feats = self.get_mmcif_features(mmcif, chain_id)
            chain_features.update(mmcif_feats)
            all_chain_features[desc] = chain_features
            sequence_features[seq] = chain_features

        all_chain_features = add_assembly_features(all_chain_features)

        # Pair and merge chain features.
        np_example = pair_and_merge(all_chain_features)

        # Pad MSA to avoid zero-sized extra_msa.
        np_example = pad_msa(np_example, 512)

        return np_example
