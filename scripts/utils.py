import argparse
from datetime import date


def add_data_args_(parser: argparse.ArgumentParser):
    parser.add_argument("--uniref90_database_path", type=str, default=None)
    parser.add_argument("--mgnify_database_path", type=str, default=None)
    parser.add_argument("--pdb70_database_path", type=str, default=None)
    parser.add_argument("--pdb_seqres_database_path", type=str, default=None)
    parser.add_argument("--uniref30_database_path", type=str, default=None)
    parser.add_argument("--uniclust30_database_path", type=str, default=None)
    parser.add_argument("--uniprot_database_path", type=str, default=None)
    parser.add_argument("--bfd_database_path", type=str, default=None)
    parser.add_argument("--jackhmmer_binary_path", type=str, default="jackhmmer")
    parser.add_argument("--hhblits_binary_path", type=str, default="hhblits")
    parser.add_argument("--hhsearch_binary_path", type=str, default="hhsearch")
    parser.add_argument("--hmmsearch_binary_path", type=str, default="hmmsearch")
    parser.add_argument("--hmmbuild_binary_path", type=str, default="hmmbuild")
    parser.add_argument("--kalign_binary_path", type=str, default="kalign")
    parser.add_argument("--max_template_date", type=str, default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--obsolete_pdbs_path", type=str, default=None)
    parser.add_argument("--release_dates_path", type=str, default=None)
