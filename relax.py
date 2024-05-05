import logging
import sys
import time

from deepfold.common import protein
from deepfold.relax import relax

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    force=True,
)
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def relax_protein(input_filepath, output_filepath, model_device="cuda"):
    """Amber relaxation."""

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

    relaxed_output_path = output_filepath
    with open(relaxed_output_path, "w") as fp:
        fp.write(struct_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")


def main(input_filepath, output_filepath):
    relax_protein(input_filepath, output_filepath)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
