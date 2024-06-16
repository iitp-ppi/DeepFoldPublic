# Copyright 2024 DeepFold Team


"""Utilities for extracting identifiers from MSA sequence desecriptions."""


import re

# Sequence coming from UniProtKB database:
_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    (?:tr|sp)
    \|
    (?P<UniqueIdentifier>[A-Za-z0-9_]+)
    \|
    (?P<EntryName>[A-Za-z0-9_]+)
    \/
    .*
    """,
    re.VERBOSE,
)

# PLM
_VDB_PATTERN = re.compile(r"^.*taxId=(?P<TaxId>[1-9][0-9]+).*")


# @dataclasses.dataclass(frozen=True)
# class Identifier:
#     tax_id: str = ""
Identifier = str


def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifier:
    """Gets species from an MSA sequence identifier."""

    matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return "uniprot:" + matches.group("EntryName").split("_")[-1]
    else:
        matches = re.search(_VDB_PATTERN, msa_sequence_identifier.strip())
        if matches:
            return "vdb:" + matches.group("TaxId")[-1]
    return Identifier()


def get_identifiers(description: str) -> Identifier:
    """Compute extra MSA features from the description."""
    return _parse_sequence_identifier(description)
