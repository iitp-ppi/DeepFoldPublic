from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Particle:
    pos: np.ndarray = field(default=np.zeros(3))
    sig: Optional[Any] = None


class CellLists:
    def __init__(
        self,
        box_size: np.ndarray,
        num_particles: int,
        cutoff_distance: float,
    ) -> None:
        self.box_size = np.array(box_size)
        self.cell_size = self.optimal_cell_size(self.box_size, num_particles, cutoff_distance)
        self.cutoff_distance = cutoff_distance
        self.num_cells = (self.box_size // self.cell_size).astype(int)
        self.cells = [[] for _ in range(np.prod(self.num_cells))]
        self.neighbor_offsets = self.calculate_neighbor_offsets()

    @staticmethod
    def optimal_cell_size(box_size, num_particles, cutoff_distance):
        # Estimate volume of simulation space
        volume = np.prod(box_size)

        # Estimate particle density
        particle_density = num_particles / volume

        # Estimate average particle spacing
        avg_spacing = (1 / particle_density) ** (1 / 3)  # Assuming 3D

        # Choose cell size slightly larger than cutoff distance
        cell_size = cutoff_distance * 1.1  # Adjust multiplier as needed

        # Ensure cell size is smaller than average particle spacing
        cell_size = np.minimum(cell_size, avg_spacing)

        return cell_size

    def clear(self):
        self.cells = [[] for _ in range(np.prod(self.num_cells))]

    def calculate_neighbor_offsets(self):
        max_offset = int(np.ceil(self.cutoff_distance / np.min(self.cell_size)))
        offsets = []
        for dz in range(-max_offset, max_offset + 1):
            for dy in range(-max_offset, max_offset + 1):
                for dx in range(-max_offset, max_offset + 1):
                    offsets.append([dx, dy, dz])

        return np.array(offsets)

    def get_index(self, pos: np.ndarray):
        pos = pos - self.box_size
        return tuple((pos // self.cell_size).astype(int) % self.num_cells)

    def add_particle(
        self,
        particle: Particle,
    ) -> None:
        index = self.get_index(particle.pos)
        self.cells[np.ravel_multi_index(index, self.num_cells)].append(particle)

    def get_neighbors(self, pos: np.ndarray) -> List[Particle]:
        index = self.get_index(pos)
        neighbor_indices = [tuple((index + offset) % self.num_cells) for offset in self.neighbor_offsets]
        neighbors = []
        for neighbor_index in neighbor_indices:
            cell_particles = self.cells[np.ravel_multi_index(neighbor_index, self.num_cells)]
            for particle in cell_particles:
                separation_squared = np.sum((pos - particle.pos) ** 2)
                if separation_squared <= self.cutoff_distance**2:
                    neighbors.append(particle)

        return neighbors
