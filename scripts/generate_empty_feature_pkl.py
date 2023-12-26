import argparse
import os
import pickle

from deepfold.model.alphafold.data.pipeline_monomer import (
    empty_template_feats,
    make_dummy_msa_feats,
    make_sequence_features,
)
from deepfold.model.alphafold.data.types import FeatureDict


def make_empty_feats(num_res: int) -> FeatureDict:
    """Generate empty input feature."""

    dummy_sequence = "A" * num_res

    sequence_features = make_sequence_features(dummy_sequence, "dummy", num_res)
    msa_features = make_dummy_msa_feats(dummy_sequence)
    template_features = empty_template_feats(num_res)

    return {
        **sequence_features,
        **msa_features,
        **template_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", "-l", type=int, required=True, help="Length of the empty sequence")
    parser.add_argument("--output_dir", "-o", type=str, default=os.getcwd(), help="Path to output directory")
    args = parser.parse_args()

    feats = make_empty_feats(args.length)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "features.pkl"), "wb") as fp:
        pickle.dump(feats, fp)


if __name__ == "__main__":
    main()
