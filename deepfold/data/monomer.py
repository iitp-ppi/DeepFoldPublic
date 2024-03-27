# Copyright 2024 DeepFold Team


import dataclasses
import functools
import os
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
def get_atom_map(can: str, mod: str) -> AtomMap:
    can = get_ligand(can)  # Canonical
    mod = get_ligand(mod)  # Modified

    # print(can.graph.edges)
    # print(mod.graph.edges)

    ismags = nx.isomorphism.ISMAGS(can.graph, mod.graph)
    subs = list(ismags.largest_common_subgraph())
    scores = []  # Heuristic
    for i, sub in enumerate(subs):
        if "CA" not in sub:
            continue
        score = 0
        for k, v in sub.items():
            if k == v:
                score += 10
            elif k[:-1] == v[:-1]:
                score += 5
            else:
                pass
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    e = subs[scores[0][0]]
    mapping = {v: k for k, v in e.items()}
    removed = set(mod.graph.nodes).difference(e.values())

    return AtomMap(mapping=mapping, removed=removed)
