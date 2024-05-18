import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from deepfold.common import protein
from deepfold.relax import relax
from deepfold.utils.log_utils import setup_logging

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def relax_protein(
    input_filepath,
    output_filepath,
    model_device="cuda",
    summary_filepath: Path | None = None,
):
    """Amber relaxation."""

    if summary_filepath is not None:
        logger.info(f"Use summary informations: {str(summary_filepath)}")
        with open(summary_filepath, "r") as fp:
            summary = json.load(fp)
        chain_index = np.asarray(summary["chain_index"])
        residue_index = np.asarray(summary["residue_index"])
        plddt = np.asarray(summary["plddt"])
    else:
        chain_index = None
        residue_index = None
        plddt = None

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=20,
        use_gpu=(model_device != "cpu"),
    )

    t = time.perf_counter()
    with open(input_filepath, "r") as fp:
        pdb_string = fp.read()
    unrelaxed_protein = protein.from_pdb_string(pdb_string)
    struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")

    prot = protein.from_relaxation(
        struct_str,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=plddt,
    )

    relaxed_output_path = output_filepath
    with open(relaxed_output_path, "w") as fp:
        fp.write(protein.to_pdb(prot))

    logger.info(f"Relaxed output written to '{relaxed_output_path}'")


def main(input_filepath, output_filepath, summary_path):
    setup_logging("relax.log")

    output_filepath = Path(output_filepath)
    if output_filepath.exists():
        raise FileExistsError(f"'{output_filepath}' exists...")

    if summary_path is not None:
        summary_filepath = Path(summary_path)
    else:
        summary_filepath = None

    relax_protein(input_filepath, output_filepath, summary_filepath=summary_filepath)


if __name__ == "__main__":
    try:
        summary_path = sys.argv[3]
    except IndexError:
        summary_path = None

    main(
        input_filepath=sys.argv[1],
        output_filepath=sys.argv[2],
        summary_path=summary_path,
    )
