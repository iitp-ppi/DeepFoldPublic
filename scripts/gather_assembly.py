import json
import sys
import warnings
from pathlib import Path

from deepfold.data.io import PDBIO
from deepfold.data.pdbx_parsing import PDBxParser, get_assemblies, get_chain_features, get_fasta, read_mmcif
from deepfold.utils.file_utils import dump_pickle


def main_pdbx(
    entry_id: str,
    debug: bool = False,
):
    entry_id = entry_id.lower()
    dv = entry_id[1:3]
    out_dir = Path("assembly") / dv / entry_id

    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    if not debug:
        warnings.filterwarnings("ignore")

    mmcif_str = read_mmcif(entry_id, mmcif_path="/gpfs/database/casp16/pdb/mmCIF")
    # mmcif_str = read_mmcif(entry_id, mmcif_path="/runs/database/pdb/mmCIF")
    parser = PDBxParser()
    o = parser.parse(mmcif_str, catch_all_errors=debug)

    # Chain mapping
    with open(out_dir / "mapping.tsv", "w") as fp:
        for x, y in o.mmcif_object.label_to_auth.items():
            line = f"{entry_id}_{x}\t{entry_id}_{y}\n"
            fp.write(line)

    assemblies = get_assemblies(o.mmcif_object)

    for v in assemblies.values():
        name = v["assembly_id"]
        with open(out_dir / f"{name}.json", "w") as fp:
            fp.write(json.dumps(v))

    with open(out_dir / "seqres.fasta", "w") as fp:
        fasta_string = get_fasta(o.mmcif_object)
        fp.write(fasta_string)

    pdb_io = PDBIO()
    for chain_id in o.mmcif_object.chain_ids:
        feats, struct = get_chain_features(
            mmcif_object=o.mmcif_object,
            model_num=1,
            chain_id=chain_id,
        )

        name = f"{entry_id}_{chain_id}"

        dump_pickle(feats, out_dir / f"{name}.pkz")

        try:
            pdb_io.set_structure(struct)
            with open(out_dir / f"{name}.pdb", "w") as fp:
                remark = f"{entry_id}_{chain_id} [auth {o.mmcif_object.label_to_auth[chain_id]}]"
                pdb_io.save(fp, remarks=remark)
        except RuntimeError as e:
            print(e)


def main():
    assert len(sys.argv) == 2
    main_pdbx(sys.argv[1])


if __name__ == "__main__":
    main()
