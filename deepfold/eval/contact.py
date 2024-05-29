from dataclasses import dataclass

import numpy as np

from deepfold.common import protein
from deepfold.eval.cell_lists import CellLists, Particle


@dataclass
class Atom:
    atom_type: int
    rid: int
    residue_index: int
    chain_index: int


def contact_map(prot: protein.Protein, cutoff: float) -> np.ndarray:
    num_res = prot.aatype.shape[0]
    num_atoms = int(np.sum(prot.atom_mask > 0.5))  # 1 then exists
    asym_id = prot.chain_index
    shape = prot.atom_positions.shape[:-1]
    mask = (1 - prot.atom_mask.astype(np.int64))[..., None].repeat(3, -1)
    masked_pos = np.ma.masked_array(prot.atom_positions, mask)
    maxi = np.ma.getdata(np.ma.max(masked_pos, axis=(0, 1)))
    mini = np.ma.getdata(np.ma.min(masked_pos, axis=(0, 1)))
    box_size = maxi - mini
    cell_lists = CellLists(box_size=box_size, num_particles=num_atoms, cutoff_distance=cutoff)

    assert asym_id is not None
    all_atom_pos = prot.atom_positions  # - mini
    for rid, at in np.ndindex(shape):
        if prot.atom_mask[rid, at] > 0.5:  # TODO: Masked array?
            cell_lists.add_particle(
                Particle(
                    pos=all_atom_pos[rid, at, :],
                    sig=Atom(
                        atom_type=at,
                        rid=rid,
                        residue_index=prot.residue_index[rid],
                        chain_index=asym_id[rid],
                    ),
                ),
            )

    contacts = [set() for _ in range(num_res)]
    for rid, at in np.ndindex(shape):
        if prot.atom_mask[rid, at] > 0.5:
            pos = all_atom_pos[rid, at, :]
            neighbor = cell_lists.get_neighbors(pos)
            for atom in neighbor:
                contacts[rid].add(atom.sig.rid)

    count = np.zeros((num_res, num_res)).astype(int)
    for i, neighbor in enumerate(contacts):
        for atom in neighbor:
            j = atom
            # if abs(i - j) <= 2 and asym_id[i] == asym_id[j]:
            # continue
            count[i, j] = 1

    return count
