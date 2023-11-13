import argparse
import os
import pickle
from typing import Mapping, Optional, Sequence, Tuple

from deepfold.data import parsers
from deepfold.model.alphafold.pipeline.make_feats import make_msa_features, make_sequence_features
from deepfold.model.alphafold.pipeline.types import FeatureDict


class Pipeline:
    """Assembles input features."""

    def __init__(self) -> None:
        pass

    def _parse_msa_data(
        self,
        alignment_dir: str,
    ) -> Mapping[str, Tuple[Sequence[str], parsers.DeletionMatrix]]:
        msa_data = {}

        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            ext = os.path.splitext(f)[-1]

            if ext == ".a3m":
                with open(path, "r") as fp:
                    m = parsers.parse_a3m(fp.read())
                data = {"msa": m.sequences, "deletion_matrix": m.deletion_matrix}
            elif ext == ".sto":
                with open(path, "r") as fp:
                    m = parsers.parse_stockholm(fp.read())
                data = {"msa": m.sequences, "deletion_matrix": m.deletion_matrix}
            else:
                continue

            msa_data[f] = data

        return msa_data

    def _get_msas(
        self,
        alignment_dir: str,
        input_sequence: Optional[str],
    ) -> Tuple[Sequence[str], parsers.DeletionMatrix]:
        msa_data = self._parse_msa_data(alignment_dir)
        if len(msa_data) == 0:
            if input_sequence is None:
                raise ValueError("If the alignment directory contains no MSAs, an input sequence must be provided")
            msa_data["dummy"] = {
                "msa": [input_sequence],
                "deletion_matrix": [[0 for _ in input_sequence]],
            }

        msas, deletion_matrices = zip(*[(v["msa"], v["deletion_matrix"]) for v in msa_data.values()])

        return msas, deletion_matrices

    def _process_msa_feats(
        self,
        alignment_dir: str,
        input_sequence: Optional[str],
    ) -> FeatureDict:
        msas, deletion_matrices = self._get_msas(alignment_dir, input_sequence)
        return make_msa_features(msas=msas, deletion_matrices=deletion_matrices)

    def process_msas(
        self,
        fasta_path: str,
        alignment_dir: str,
    ) -> FeatureDict:
        """Assemble MSA features."""
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, _ = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(f"More than one input sequence found in '{fasta_path}'")
        input_sequence = input_seqs[0]
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(input_sequence, "", num_res)
        msa_features = self._process_msa_feats(alignment_dir, input_sequence)

        return {
            **sequence_features,
            **msa_features,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta_path", type=str, help="Path to a FASTA file")
    parser.add_argument("--alignment_dir", type=str, default=None, help="Path to alignment directory")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="Path to output directory")
    args = parser.parse_args()

    feats = Pipeline().process_msas(fasta_path=args.fasta_path, alignment_dir=args.alignment_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "features.pkl"), "wb") as fp:
        pickle.dump(feats, fp)


if __name__ == "__main__":
    main()
