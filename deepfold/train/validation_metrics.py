from typing import Dict, List, Optional, Set

import torch

from deepfold.common import residue_constants as rc
from deepfold.losses.confidence import lddt_ca
from deepfold.losses.geometry import superimpose

VALIDATION_METRICS_NAMES = {"lddt_ca", "drmsd_ca", "alignment_rmsd", "gdt_ts", "gdt_ha"}


def compute_validation_metrics(
    predicted_atom_positions: torch.Tensor,
    target_atom_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    metrics_names: Set[str],
) -> Dict[str, torch.Tensor]:
    val_metrics = {}

    assert isinstance(metrics_names, set)
    if len(metrics_names) == 0:
        raise ValueError(f"Validation `metrics_names` set is empty. VALIDATION_METRICS_NAMES='{VALIDATION_METRICS_NAMES}'")
    assert metrics_names.issubset(VALIDATION_METRICS_NAMES)

    pred_coords = predicted_atom_positions
    gt_coords = target_atom_positions
    all_atom_mask = atom_mask

    if "lddt_ca" in metrics_names:
        val_metrics["lddt_ca"] = lddt_ca(
            all_atom_pred_pos=pred_coords,
            all_atom_positions=gt_coords,
            all_atom_mask=all_atom_mask,
            eps=1e-8,
            per_residue=False,
        )
        if metrics_names == {"lddt_ca"}:
            return val_metrics

    gt_coords_masked = gt_coords * all_atom_mask[..., None]
    pred_coords_masked = pred_coords * all_atom_mask[..., None]
    ca_pos = rc.atom_order["CA"]
    gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]
    pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]
    all_atom_mask_ca = all_atom_mask[..., ca_pos]

    if "drmsd_ca" in metrics_names:
        val_metrics["drmsd_ca"] = drmsd(
            structure_1=pred_coords_masked_ca,
            structure_2=gt_coords_masked_ca,
            mask=all_atom_mask_ca,  # still required here to compute n
        )

    superimposition_metric_names = {"alignment_rmsd", "gdt_ts", "gdt_ha"} & metrics_names

    if superimposition_metric_names:
        superimposed_pred, alignment_rmsd = superimpose(
            tgt_atoms=gt_coords_masked_ca,
            src_atoms=pred_coords_masked_ca,
            mask=all_atom_mask_ca,
        )

        if "alignment_rmsd" in metrics_names:
            val_metrics["alignment_rmsd"] = alignment_rmsd

        if "gdt_ts" in metrics_names:
            val_metrics["gdt_ts"] = gdt_ts(
                p1=superimposed_pred,
                p2=gt_coords_masked_ca,
                mask=all_atom_mask_ca,
            )

        if "gdt_ha" in metrics_names:
            val_metrics["gdt_ha"] = gdt_ha(
                p1=superimposed_pred,
                p2=gt_coords_masked_ca,
                mask=all_atom_mask_ca,
            )

    return val_metrics


def drmsd(
    structure_1: torch.Tensor,
    structure_2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    def prep_d(structure):
        d = structure[..., :, None, :] - structure[..., None, :, :]
        d = d**2
        d = torch.sqrt(torch.sum(d, dim=-1))
        return d

    d1 = prep_d(structure_1)
    d2 = prep_d(structure_2)

    drmsd = d1 - d2
    drmsd = drmsd**2
    if mask is not None:
        drmsd = drmsd * (mask[..., None] * mask[..., None, :])
    drmsd = torch.sum(drmsd, dim=(-1, -2))
    n = d1.shape[-1] if mask is None else torch.sum(mask, dim=-1)
    drmsd = drmsd * (1 / (n * (n - 1))) if n > 1 else (drmsd * 0.0)
    drmsd = torch.sqrt(drmsd)

    return drmsd


def gdt(
    p1: torch.Tensor,
    p2: torch.Tensor,
    mask: torch.Tensor,
    cutoffs: List[float],
) -> torch.Tensor:
    p1 = p1.float()
    p2 = p2.float()
    n = torch.sum(mask, dim=-1)
    distances = torch.sqrt(torch.sum((p1 - p2) ** 2, dim=-1))
    scores = []
    for c in cutoffs:
        score = torch.sum((distances <= c) * mask, dim=-1) / n
        score = torch.mean(score)
        scores.append(score)
    return sum(scores) / len(scores)


def gdt_ts(
    p1: torch.Tensor,
    p2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return gdt(p1, p2, mask, cutoffs=[1.0, 2.0, 4.0, 8.0])


def gdt_ha(
    p1: torch.Tensor,
    p2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return gdt(p1, p2, mask, cutoffs=[0.5, 1.0, 2.0, 4.0])
