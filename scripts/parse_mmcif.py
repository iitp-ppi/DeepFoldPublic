import json
import logging
import sys
import warnings
from pathlib import Path

import tqdm

from deepfold.data.pdbx_parsing import MMCIFParser, get_assembly_infos, get_chain_features, get_fasta, read_mmcif
from deepfold.utils.file_utils import dump_pickle

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def main_pdbx(entry_id: str):
    entry_id = entry_id.upper()

    out_dir = Path("out") / entry_id[1:3] / entry_id
    if (out_dir / f"{entry_id}.fasta").exists():
        return

    mmcif_str = read_mmcif(entry_id, mmcif_path="./mmCIF")
    parser = MMCIFParser()
    o = parser.parse(mmcif_str, entry_id=entry_id)

    out_dir.mkdir(parents=True, exist_ok=True)

    for model_num, chains in o.mmcif_object.models.items():
        for chain_id in chains.keys():
            feats = get_chain_features(o.mmcif_object, model_num, chain_id)
            dump_pickle(feats, out_dir / f"{entry_id}_{chain_id}.{model_num}.pkl")

    assems = get_assembly_infos(o.mmcif_object)
    for name, assem in assems.items():
        assem_str = json.dumps(assem)
        with open(out_dir / f"{name}.json", "w") as fp:
            fp.write(assem_str)

    fasta_str = get_fasta(o.mmcif_object)
    with open(out_dir / f"{entry_id}.fasta", "w") as fp:
        fp.write(fasta_str)


def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    input_file = sys.argv[1]

    with open(input_file, "r") as fp:
        lines = fp.read().split("\n")

    ch = logging.FileHandler(f"{input_file}.err")
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    with open(sys.argv[2], "a") as fp:
        for line in (pbar := tqdm.tqdm(lines)):
            line = line.strip()
            pbar.set_description_str(f"PDBx: {line}")
            if line == "":
                continue
            try:
                main_pdbx(line)
            except Exception:
                # raise
                fp.write(f"{line}\n")
                continue
            except KeyboardInterrupt:
                fp.write(f"{line}\n")
                continue


if __name__ == "__main__":
    main()
