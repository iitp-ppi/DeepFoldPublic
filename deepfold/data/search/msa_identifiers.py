# Copyright 2024 DeepFold Team


"""Utilities for extracting identifiers from MSA sequence desecriptions."""


import dataclasses
import re

# Sequence coming from UniProtKB database:
_UNIPROT_PATTERN = re.compile(
    r"""
    ^
    (?:tr|sp)
    \|
    (?P<UniqueIdentifier>[A-Za-z0-9_]+)
    \|
    (?P<EntryName>[A-Za-z0-9_]+)\s
    (?P<ProteinName>.+)\s
    OS=(?P<OrganismName>.+)\s
    OX=(?P<OrganismIdentifier>[1-9][0-9]*)\s
    (GN=(?P<GeneName>.+)\s)?
    PE=([1-9][0-9]*)\s
    SV=([1-9][0-9]*)$
    """,
    re.VERBOSE,
)


@dataclasses.dataclass(frozen=True)
class Identifier:
    tax_id: str = ""


def _parse_sequence_identifier_uniprot(msa_sequence_identifier: str) -> Identifier:
    """Gets species from an MSA sequence identifier."""

    matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return Identifier(tax_id=matches.group("OrganismIdentifier"))
    return Identifier()


def get_uniprot_identifierss(description: str) -> Identifier:
    """Compute extra MSA features from the description."""

    sequence_identifier = _extract_sequence_identifier(description)
    if sequence_identifier is None:
        return Identifier()
    return _parse_sequence_identifier_uniprot(sequence_identifier)


_UNIREF_PATTERN = re.compile(
    r"""
    ^
    (?P<UniqueIdentifier>[A-Za-z0-9_]+)\s+
    (?P<ClusterName>.+)\s+
    n=(?P<Members>[1-9][0-9]*)\s+
    Tax=(?P<TaxonName>.+)\s+
    TaxID=(?P<TaxonIdentifier>[1-9][0-9]*)\s+
    RepID=(?P<RepresentativeMember>.+)$
    """,
    re.VERBOSE,
)


def _parse_sequence_identifier_uniref(msa_sequence_identifier: str) -> Identifier:
    """Gets species from an MSA sequence identifier."""

    matches = re.search(_UNIREF_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return Identifier(tax_id=matches.group("TaxonIdentifier"))
    return Identifier()


def _extract_sequence_identifier(description: str) -> str | None:
    """Extracct sequence identifier from description. Returns None if no match."""

    split_description = description.split()
    if split_description:
        return split_description[0].partition("/")[0]
    return None


def get_uniref_identifierss(description: str) -> Identifier:
    """Compute extra MSA features from the description."""

    sequence_identifier = _extract_sequence_identifier(description)
    if sequence_identifier is None:
        return Identifier()
    return _parse_sequence_identifier_uniref(sequence_identifier)
