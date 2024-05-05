from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from deepfold.common import residue_constants as rc
from deepfold.utils.tensor_utils import array_tree_map, masked_mean, tensor_tree_map


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    tolerance_factor_soft: float = 12.0,
    tolerance_factor_hard: float = 12.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    equation 44 & 45 (Supplementary '1.9.11 Structural violations').

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation present.

    """
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == rc.resname_to_idx["PRO"]
    gt_length = (~next_is_proline) * rc.between_res_bond_length_c_n[0] + next_is_proline * rc.between_res_bond_length_c_n[1]
    gt_stddev = (~next_is_proline) * rc.between_res_bond_length_stddev_c_n[0] + next_is_proline * rc.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.relu(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1))
    n_ca_bond_length = torch.sqrt(eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_ca_c_n[0]
    gt_stddev = rc.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(eps + (ca_c_n_cos_angle - gt_angle) ** 2)
    ca_c_n_loss_per_residue = torch.relu(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_c_n_ca[0]
    gt_stddev = rc.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(eps + torch.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = torch.relu(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both neighbouring residues).
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (F.pad(per_residue_loss_sum, (0, 1)) + F.pad(per_residue_loss_sum, (1, 0)))

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [
                c_n_violation_mask,
                ca_c_n_violation_mask,
                c_n_ca_violation_mask,
            ],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        F.pad(violation_mask, (0, 1)),
        F.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft: float = 1.5,
    overlap_tolerance_hard: float = 1.5,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of equation 46 (Supplementary '1.9.11 Structural violations').

    Args:
      atom14_pred_positions: Predicted positions of atoms in global prediction frame.
      atom14_atom_exists: Mask denoting whether atom at positions exists for given amino acid type.
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom shape (N, 14)

    """
    dtype = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_pred_positions[..., :, None, :, None, :] - atom14_pred_positions[..., None, :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (atom14_atom_exists[..., :, None, :, None] * atom14_atom_exists[..., None, :, None, :]).type(dtype)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (residue_index[..., :, None, None, None] < residue_index[..., None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = F.one_hot(residue_index.new_tensor(2), num_classes=14)
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])),
        *c_one_hot.shape,
    )
    c_one_hot = c_one_hot.type(dtype)
    n_one_hot = F.one_hot(residue_index.new_tensor(0), num_classes=14)
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])),
        *n_one_hot.shape,
    )
    n_one_hot = n_one_hot.type(dtype)

    neighbour_mask = (residue_index[..., :, None, None, None] + 1) == residue_index[..., None, :, None, None]
    c_n_bonds = neighbour_mask * c_one_hot[..., None, None, :, None] * n_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(*((1,) * len(residue_index.shape[:-1])), 1).squeeze(-1)
    cys_sg_one_hot = F.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = cys_sg_one_hot[..., None, None, :, None] * cys_sg_one_hot[..., None, None, None, :]
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (atom14_atom_radius[..., :, None, :, None] + atom14_atom_radius[..., None, :, None, :])

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.relu(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(dists_to_low_error, axis=(-3, -1))

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss: float = 0.0,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with the same residues of
    equation 46 (Supplementary '1.9.11 Structural violations').

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape

    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])),
        *dists_masks.shape,
    )
    dists_masks = atom14_atom_exists[..., :, :, None] * atom14_atom_exists[..., :, None, :] * dists_masks

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (atom14_pred_positions[..., :, :, None, :] - atom14_pred_positions[..., :, None, :, :]) ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.relu(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = torch.relu(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * ((dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound))

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0],
        torch.max(violations, axis=-1)[0],
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [rc.van_der_waals_radius[name[0]] for name in rc.atom_types]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = batch["atom14_atom_exists"] * atomtype_radius[batch["residx_atom14_to_atom37"]]

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(restype_atom14_bounds["lower_bound"])[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(restype_atom14_bounds["upper_bound"])[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(between_residue_clashes["per_atom_clash_mask"], dim=-1)[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations["ca_c_n_loss_mean"],  # ()
            "angles_c_n_ca_loss_mean": connection_violations["c_n_ca_loss_mean"],  # ()
            "connections_per_residue_loss_sum": connection_violations["per_residue_loss_sum"],  # (N)
            "connections_per_residue_violation_mask": connection_violations["per_residue_violation_mask"],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes["per_atom_loss_sum"],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes["per_atom_clash_mask"],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations["per_atom_loss_sum"],  # (N, 14)
            "per_atom_violations": residue_violations["per_atom_violations"],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
) -> Dict[str, np.ndarray]:
    violations = find_structural_violations(
        batch=array_tree_map(fn=torch.tensor, tree=batch),
        atom14_pred_positions=torch.tensor(atom14_pred_positions),
        violation_tolerance_factor=violation_tolerance_factor,
        clash_overlap_tolerance=clash_overlap_tolerance,
    )
    violations = tensor_tree_map(fn=np.array, tree=violations)
    return violations


def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.

    Returns:
      Fraction of consecutive CA-CA pairs with violation.

    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1))
    violations = (ca_ca_distance - rc.ca_ca) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    return masked_mean(mask, violations, -1)


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    violation_metrics = {}
    violation_metrics["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    violation_metrics["violations_between_residue_bond"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["between_residues"]["connections_per_residue_violation_mask"],
        dim=-1,
    )
    violation_metrics["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    violation_metrics["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"],
            dim=-1,
        )[0],
        dim=-1,
    )
    violation_metrics["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return violation_metrics


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    violation_metrics = compute_violation_metrics(
        batch=array_tree_map(fn=torch.tensor, tree=batch),
        atom14_pred_positions=torch.tensor(atom14_pred_positions),
        violations=array_tree_map(fn=torch.tensor, tree=violations),
    )
    violation_metrics = tensor_tree_map(fn=lambda t: t.numpy(), tree=violation_metrics)
    return violation_metrics


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists, dim=(-1, -2))

    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"] + violations["within_residues"]["per_atom_loss_sum"],
        dim=(-1, -2),
    )
    l_clash = l_clash / (eps + num_atoms)

    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )

    return loss
