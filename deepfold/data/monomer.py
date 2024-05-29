# Copyright 2024 DeepFold Team


import dataclasses
import functools
import os
import re
from typing import Dict, Set

import networkx as nx
import requests


@functools.lru_cache(maxsize=24)
def fetch_ligand(id: str) -> str:
    ligand_id = id.upper()

    r = requests.get(f"https://files.wwpdb.org/pub/pdb/data/monomers/{ligand_id}")
    if r.status_code == 200:
        text = r.text
        return text
    RuntimeError(f"Cannot fetch '{ligand_id}'")


@functools.lru_cache(maxsize=24)
def read_ligand(
    ligand_id: str,
    monomer_path: os.PathLike | None = None,
) -> str:
    ligand_id = ligand_id.upper()
    monomer_path = "." if monomer_path is None else monomer_path
    ligand_path = os.path.join(monomer_path, ligand_id)
    if monomer_path is not None and os.path.exists(ligand_path):
        with open(ligand_path, "r") as fp:
            return fp.read()
    else:
        return fetch_ligand(ligand_id)


@dataclasses.dataclass
class Ligand:
    id: str = ""
    name: str = ""
    graph: nx.Graph = dataclasses.field(default=nx.Graph())


@functools.lru_cache(maxsize=24)
def get_ligand(
    ligand_id: str,
    noh: bool = False,
    monomer_path: os.PathLike | None = None,
) -> Ligand:
    text = read_ligand(ligand_id, monomer_path=monomer_path)
    name = ""
    graph = nx.Graph()

    for s in text.split("\n"):
        entry = s.split()
        if len(entry) == 0:
            continue
        header = entry[0]
        entry = entry[1:]

        if header == "HET":
            ligand_id = entry[0]
        elif header == "HETNAM":
            name = entry[1]
        elif header == "CONECT":
            e1 = entry[0]
            if noh and e1.startswith("H"):
                continue
            end = int(entry[1]) + 2
            for e2 in entry[2:end]:
                if noh and e2.startswith("H"):
                    continue
                graph.add_edge(e1, e2)
        else:
            pass

    return Ligand(id=ligand_id, name=name, graph=graph)


@dataclasses.dataclass(frozen=True)
class AtomMap:
    mapping: Dict[str, str]
    removed: Set[str]


@functools.lru_cache(maxsize=128)
def build_atom_map(can: str, mod: str) -> AtomMap:
    can_lig = get_ligand(can, noh=True)  # Canonical
    mod_lig = get_ligand(mod, noh=True)  # Modified

    cutoff = {
        "ALA": 2,
        "ARG": 6,
        "ASN": 3,
        "ASP": 3,
        "CYS": 2,
        "GLN": 4,
        "GLU": 4,
        "GLY": 2,
        "HIS": 4,
        "ILE": 3,
        "LEU": 3,
        "LYS": 5,
        "MET": 4,
        "PHE": 5,
        "PRO": 2,
        "SER": 2,
        "THR": 2,
        "TRP": 6,
        "TYR": 6,
        "VAL": 2,
    }

    k = cutoff[can]  # How far from CA?
    shortest_paths = nx.single_source_shortest_path_length(mod_lig.graph, "CA")
    nodes_to_remove = [node for node, dist in shortest_paths.items() if dist > k]
    mod_lig.graph.remove_nodes_from(nodes_to_remove)

    scores = []  # Heuristic
    ismags = nx.isomorphism.ISMAGS(can_lig.graph, mod_lig.graph)
    largest_common_subgraph = list(ismags.largest_common_subgraph())
    for i, sub in enumerate(largest_common_subgraph):
        if "CA" not in sub:
            continue
        score = 0
        for k, v in sub.items():
            if k == v:
                score += 10
            elif k[0] != v[0]:
                score -= 100
            else:
                pass
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    e = largest_common_subgraph[scores[0][0]]
    mapping = {v: k for k, v in e.items() if k[0] == v[0]}
    removed = set(mod_lig.graph.nodes).difference(mapping.keys())

    return AtomMap(mapping=mapping, removed=removed)
