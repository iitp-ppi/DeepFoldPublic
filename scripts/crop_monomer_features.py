import argparse
from pathlib import Path

from deepfold.utils.feats_utils import crop_features
from deepfold.utils.file_utils import dump_pickle, load_pickle


def main(args: argparse.Namespace) -> None:
    feats = load_pickle(args.input_filepath)
    i, j = map(int, args.interval.split("-"))
    new_feats = crop_features(feats, i, j)
    if not args.force and args.output_filepath.exists():
        raise FileExistsError(f"File already exists: {str(args.output_filepath)}")
    args.output_filepath.parent.mkdir(parents=True, exist_ok=True)
    dump_pickle(new_feats, args.output_filepath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        type=Path,
        required=True,
        dest="input_filepath",
    )
    parser.add_argument(
        "-o",
        type=Path,
        required=True,
        dest="output_filepath",
    )
    parser.add_argument(
        "-r",
        type=str,
        required=True,
        dest="interval",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    args = parser.parse_args()

    for k, v in vars(args).items():
        if isinstance(v, list):
            for i, x in enumerate(v):
                print(f"{k}[{i}]={x}")
        else:
            print(f"{k}={v}")

    return args


if __name__ == "__main__":
    main(parse_args())
    exit(0)
