import logging
import os
from multiprocessing import cpu_count
from typing import Dict, Optional, Union

import numpy as np

from deepfold.search import parsers
from deepfold.search.tools import hhblits, hhsearch, hmmsearch, jackhmmer

TemplateSearcher = Union[hhsearch.HHSearch, hmmsearch.Hmmsearch]


logger = logging.getLogger(__name__)


class AlignmentRunner:
    """Runs alignment tools and saves the results"""

    def __init__(
        self,
        template_searcher: Optional[TemplateSearcher] = None,
        jackhmmer_binary_path: Optional[str] = None,
        hhblits_binary_path: Optional[str] = None,
        uniref90_database_path: Optional[str] = None,
        mgnify_database_path: Optional[str] = None,
        bfd_database_path: Optional[str] = None,
        uniclust30_database_path: Optional[str] = None,
        uniprot_database_path: Optional[str] = None,
        use_small_bfd: bool = False,
        uniref_max_hits: int = 10000,
        mgnify_max_hits: int = 5000,
        uniprot_max_hits: int = 50000,
        num_cpus: Optional[int] = None,
    ):
        self.template_searcher = template_searcher

        db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                    uniprot_database_path,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
        }

        for name, dic in db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if binary is None and not all([x is None for x in dbs]):
                raise ValueError(f"{name} DBs provided but {name} binary is None")

        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.uniprot_max_hits = uniprot_max_hits
        self.use_small_bfd = use_small_bfd

        if num_cpus is None:
            num_cpus = cpu_count()

        self.jackhmmer_uniref90_runner = None
        if jackhmmer_binary_path is not None and uniref90_database_path is not None:
            self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniref90_database_path,
                n_cpu=num_cpus,
            )

        self.jackhmmer_small_bfd_runner = None
        self.hhblits_bfd_uniclust_runner = None
        if bfd_database_path is not None:
            if use_small_bfd:
                self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=bfd_database_path,
                    n_cpu=num_cpus,
                )
            else:
                dbs = [bfd_database_path]
                if uniclust30_database_path is not None:
                    dbs.append(uniclust30_database_path)
                if uniclust30_database_path is not None:
                    dbs.append(uniclust30_database_path)
                self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=dbs,
                    n_cpu=num_cpus,
                )

        self.jackhmmer_mgnify_runner = None
        if mgnify_database_path is not None:
            self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=mgnify_database_path,
                n_cpu=num_cpus,
            )

        self.jackhmmer_uniprot_runner = None
        if uniprot_database_path is not None:
            self.jackhmmer_uniprot_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniprot_database_path,
                n_cpu=num_cpus,
            )

        self.template_searcher = template_searcher

    def run(
        self,
        fasta_path: str,
        output_dir: str,
    ):
        """Runs alignment tools on a sequence"""

        if self.jackhmmer_uniref90_runner is not None:
            uniref90_out_path = os.path.join(output_dir, "uniref90_hits.sto")

            jackhmmer_uniref90_result = run_msa_tool(
                msa_runner=self.jackhmmer_uniref90_runner,
                fasta_path=fasta_path,
                msa_out_path=uniref90_out_path,
                msa_format="sto",
                max_sto_sequences=self.uniref_max_hits,
            )

            template_msa = jackhmmer_uniref90_result["sto"]
            template_msa = parsers.deduplicate_stockholm_msa(template_msa)
            template_msa = parsers.remove_empty_columns_from_stockholm_msa(template_msa)

            if self.template_searcher is not None:
                if self.template_searcher.input_format == "sto":
                    _ = self.template_searcher.query(template_msa, output_dir=output_dir)
                elif self.template_searcher.input_format == "a3m":
                    uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(template_msa)
                    _ = self.template_searcher.query(uniref90_msa_as_a3m, output_dir=output_dir)
                else:
                    fmt = self.template_searcher.input_format
                    raise ValueError(f"Unrecognized template input format: {fmt}")

        if self.jackhmmer_mgnify_runner is not None:
            mgnify_out_path = os.path.join(output_dir, "mgnify_hits.sto")
            run_msa_tool(
                msa_runner=self.jackhmmer_mgnify_runner,
                fasta_path=fasta_path,
                msa_out_path=mgnify_out_path,
                msa_format="sto",
                max_sto_sequences=self.mgnify_max_hits,
            )

        if self.use_small_bfd and self.jackhmmer_small_bfd_runner is not None:
            bfd_out_path = os.path.join(output_dir, "small_bfd_hits.sto")
            run_msa_tool(
                msa_runner=self.jackhmmer_small_bfd_runner,
                fasta_path=fasta_path,
                msa_out_path=bfd_out_path,
                msa_format="sto",
            )
        elif self.hhblits_bfd_uniclust_runner is not None:
            uni_name = "uni"
            for db_name in self.hhblits_bfd_uniclust_runner.databases:
                if "uniref" in db_name.lower():
                    uni_name = f"{uni_name}ref"
                elif "uniclust" in db_name.lower():
                    uni_name = f"{uni_name}clust"

            bfd_out_path = os.path.join(output_dir, f"bfd_{uni_name}_hits.a3m")
            run_msa_tool(
                msa_runner=self.hhblits_bfd_uniclust_runner,
                fasta_path=fasta_path,
                msa_out_path=bfd_out_path,
                msa_format="a3m",
            )

        if self.jackhmmer_uniprot_runner is not None:
            uniprot_out_path = os.path.join(output_dir, "uniprot_hits.sto")
            run_msa_tool(
                self.jackhmmer_uniprot_runner,
                fasta_path=fasta_path,
                msa_out_path=uniprot_out_path,
                msa_format="sto",
                max_sto_sequences=self.uniprot_max_hits,
            )


def run_msa_tool(
    msa_runner,
    fasta_path: str,
    msa_out_path: str,
    msa_format: str,
    max_sto_sequences: Optional[int] = None,
) -> Dict[str, parsers.MSA]:
    """Runs an MSA tool, checking if output already exists first."""
    if msa_format == "sto" and max_sto_sequences is not None:
        result = msa_runner.query(fasta_path, max_sto_sequences)[0]
    else:
        result = msa_runner.query(fasta_path)[0]

    assert msa_out_path.split(".")[-1] == msa_format
    with open(msa_out_path, "w") as f:
        f.write(result[msa_format])

    return result
