import argparse
import concurrent
import logging
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor

from deepfold.data import parsers

logger = logging.getLogger(__name__)


def parse_stockholm_file(alignment_dir: str, stockholm_file: str):
    path = os.path.join(alignment_dir, stockholm_file)
    file_name, _ = os.path.splitext(stockholm_file)
    with open(path, "r") as infile:
        msa = parsers.parse_stockholm(infile.read())
        infile.close()
    return {file_name: msa}


def parse_a3m_file(alignment_dir: str, a3m_file: str):
    path = os.path.join(alignment_dir, a3m_file)
    file_name, _ = os.path.splitext(a3m_file)
    with open(path, "r") as infile:
        msa = parsers.parse_a3m(infile.read())
        infile.close()
    return {file_name: msa}


def run_parse_all_msa_files_multiprocessing(stockholm_files: list, a3m_files: list, alignment_dir: str):
    # Number of workers based on the tasks
    msa_results = {}
    a3m_tasks = [(alignment_dir, f) for f in a3m_files]
    sto_tasks = [(alignment_dir, f) for f in stockholm_files]
    with ProcessPoolExecutor(max_workers=len(a3m_tasks) + len(sto_tasks)) as executor:
        a3m_futures = {executor.submit(parse_a3m_file, *task): task for task in a3m_tasks}
        sto_futures = {executor.submit(parse_stockholm_file, *task): task for task in sto_tasks}

        for future in concurrent.futures.as_completed(a3m_futures | sto_futures):
            try:
                result = future.result()
                msa_results.update(result)
            except Exception as exc:
                logger.error(f"Task generated an exception: {exc}")
        return msa_results


def main():
    parser = argparse.ArgumentParser(description="Process msa files in parallel")
    parser.add_argument("--alignment_dir", type=str, help="path to alignment dir")
    args = parser.parse_args()
    alignment_dir = args.alignment_dir
    stockholm_files = [i for i in os.listdir(alignment_dir) if all([i.endswith(".sto"), "hmm_output" not in i, "uniprot_hits" not in i])]
    a3m_files = [i for i in os.listdir(alignment_dir) if i.endswith(".a3m")]
    msa_data = run_parse_all_msa_files_multiprocessing(stockholm_files, a3m_files, alignment_dir)

    logging.info(f"MSA parsed from '{alignment_dir}'")


if __name__ == "__main__":
    main()
