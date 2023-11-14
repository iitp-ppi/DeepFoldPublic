import argparse
import os
import pickle
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

import deepfold.common.protein as protein
import deepfold.common.residue_constants as rc
from deepfold.data import parsers
from deepfold.model.alphafold.pipeline.types import FeatureDict


def _pdb_to_template(pdb_str: str, _zero_center_positions: bool = False) -> FeatureDict:
    """Parse PDB to atom37 format array."""

    prot = protein.from_pdb_string(pdb_str)

    all_atom_positions = prot.atom_positions
    binary_mask = prot.atom_mask.astype(bool)
    trans_vec = all_atom_positions[binary_mask].mean(axis=0)
    all_atom_positions[binary_mask] -= trans_vec

    entry = {
        "template_aatype": prot.aatype,
        "template_all_atom_positions": all_atom_positions,
        "template_all_atom_mask": prot.atom_mask,
        "template_sum_probs": np.ma.masked_array(prot.b_factors, mask=binary_mask).mean(axis=-1, keepdims=True).data,
    }
    return entry


def _pad_dict(entry: FeatureDict, start: int, res_num: int) -> FeatureDict:
    """Pad entries with start position and number of residues."""

    for k, v in entry.items():
        pad_width = [[0, 0] for _ in range(v.ndim)]
        pad_width[0][0] = start - 1
        pad_width[0][1] = res_num - v.shape[0] - start + 1
        entry[k] = np.pad(v, pad_width, mode="constant", constant_values=0.0)

    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_pkl", "-pkl", type=str, help="Path to a feature pickle")
    parser.add_argument("--output_dir", "-o", type=str, default=os.getcwd(), help="Path to output directory")
    parser.add_argument(
        "--domain_pdbs",
        "-f",
        type=str,
        action="store",
        dest="pdbs",
        nargs="*",
        default=[""],
        help="Path to domain pdb files",
    )
    args = parser.parse_args()

    with open(args.feature_pkl, "rb") as fp:
        feats = pickle.load(fp)

    # Get number of residues from aatype entry
    res_num = feats["aatype"].shape[-2]

    pdbs = []
    for s in args.pdbs:
        # Cut string to path and starting position
        sp = s.split(":")
        start = int(sp[-1])
        path = "".join(sp[:-1])

        with open(path, "r") as fp:
            pdb_str = fp.read()

        entry = _pdb_to_template(pdb_str)
        entry = _pad_dict(entry, start, res_num)

        pdbs.append(entry)

    template_feats = {
        k: []
        for k in [
            "template_aatype",
            "template_all_atom_positions",
            "template_all_atom_mask",
            "template_sum_probs",
        ]
    }
    for d in pdbs:
        for k, v in d.items():
            template_feats[k].append(v[None, ...])
    for k, v in template_feats.items():
        template_feats[k] = np.concatenate(v, axis=0)

    feats = dict(feats, **template_feats)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "features.pkl"), "wb") as fp:
        pickle.dump(feats, fp)


if __name__ == "__main__":
    main()
