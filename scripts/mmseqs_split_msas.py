import logging
from argparse import ArgumentParser
from pathlib import Path

logger = logging.getLogger(__name__)


def split_msa(merged_msa: Path, output_folder: Path):
    with merged_msa.open("r") as f:
        line = f.readline()
        msa = [line.strip().replace("\0", "")]

        while line:
            if line.count("\0") > 0:
                filename = msa[0][1:].split(" ")[0].strip().replace("/", "_").replace(">", "") + ".a3m"
                output_folder.joinpath(filename).write_text("\n".join(msa))
                msa = []

                line = line.replace("\0", "")

            msa.append(line.strip())
            line = f.readline()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    parser = ArgumentParser()
    parser.add_argument(
        "input_a3m",
        help="The search folder in which you ran colabfold_search with the final.a3m",
    )
    parser.add_argument(
        "output_folder",
        help="Will contain all the a3m files",
    )
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True)

    logger.info("Splitting MSAs")
    split_msa(Path(args.input_a3m), output_folder)
    logger.info("Done")


if __name__ == "__main__":
    main()
