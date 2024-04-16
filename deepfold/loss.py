import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfold.common import residue_constants as rc
from deepfold.config import LossConfig
from deepfold.losses.auxillary import experimentally_resolved_loss
from deepfold.losses.confidence import plddt_loss, tm_loss
from deepfold.losses.geometry import compute_renamed_ground_truth, distogram_loss, fape_loss, supervised_chi_loss
from deepfold.losses.masked_msa import masked_msa_loss
from deepfold.losses.violation import find_structural_violations, violation_loss
from deepfold.utils.rigid_utils import Rigid, Rotation
from deepfold.utils.tensor_utils import array_tree_map, tensor_tree_map

logger = logging.getLogger(__name__)


class AlphaFoldLoss(nn.Module):
    """AlphaFold loss module.

    Supplementary '1.9 Loss functions and auxiliary heads'.

    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.fape_loss_config = config.fape_loss_config
        self.supervised_chi_loss_config = config.supervised_chi_loss_config
        self.distogram_loss_config = config.distogram_loss_config
        self.masked_msa_loss_config = config.masked_msa_loss_config
        self.plddt_loss_config = config.plddt_loss_config
        self.experimentally_resolved_loss_config = config.experimentally_resolved_loss_config
        self.violation_loss_config = config.violation_loss_config
        self.tm_loss_config = config.tm_loss_config

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """AlphaFold loss forward pass.

        Args:
            outputs: forward pass output dict
            batch: train batch dict

        Returns:
            scaled_weight_total_loss: total loss connected to the  graph
            losses: dict with loss detached from the graph

        """
        batch_size = batch["aatype"].size(0)

        if "violations" not in outputs.keys():
            outputs["violations"] = find_structural_violations(
                batch=batch,
                atom14_pred_positions=outputs["sm_positions"][:, -1],
                violation_tolerance_factor=self.violation_loss_config.violation_tolerance_factor,
                clash_overlap_tolerance=self.violation_loss_config.clash_overlap_tolerance,
            )

        if "renamed_atom14_gt_positions" not in outputs.keys():
            batch.update(compute_renamed_ground_truth(batch=batch, atom14_pred_positions=outputs["sm_positions"][:, -1]))

        losses = {}

        losses["fape"] = fape_loss(
            outputs=outputs,
            batch=batch,
            backbone_clamp_distance=self.fape_loss_config.backbone_clamp_distance,
            backbone_loss_unit_distance=self.fape_loss_config.backbone_loss_unit_distance,
            backbone_weight=self.fape_loss_config.backbone_weight,
            sidechain_clamp_distance=self.fape_loss_config.sidechain_clamp_distance,
            sidechain_length_scale=self.fape_loss_config.sidechain_length_scale,
            sidechain_weight=self.fape_loss_config.sidechain_weight,
            eps=self.fape_loss_config.eps,
        )

        losses["supervised_chi"] = supervised_chi_loss(
            angles_sin_cos=outputs["sm_angles"],
            unnormalized_angles_sin_cos=outputs["sm_unnormalized_angles"],
            aatype=batch["aatype"],
            seq_mask=batch["seq_mask"],
            chi_mask=batch["chi_mask"],
            chi_angles_sin_cos=batch["chi_angles_sin_cos"],
            chi_weight=self.supervised_chi_loss_config.chi_weight,
            angle_norm_weight=self.supervised_chi_loss_config.angle_norm_weight,
            eps=self.supervised_chi_loss_config.eps,
        )

        losses["distogram"] = distogram_loss(
            logits=outputs["distogram_logits"],
            pseudo_beta=batch["pseudo_beta"],
            pseudo_beta_mask=batch["pseudo_beta_mask"],
            min_bin=self.distogram_loss_config.min_bin,
            max_bin=self.distogram_loss_config.max_bin,
            num_bins=self.distogram_loss_config.num_bins,
            eps=self.distogram_loss_config.eps,
        )

        losses["masked_msa"] = masked_msa_loss(
            logits=outputs["masked_msa_logits"],
            true_msa=batch["true_msa"],
            bert_mask=batch["bert_mask"],
            eps=self.masked_msa_loss_config.eps,
        )

        losses["plddt_loss"] = plddt_loss(
            logits=outputs["lddt_logits"],
            all_atom_pred_pos=outputs["final_atom_positions"],
            all_atom_positions=batch["all_atom_positions"],
            all_atom_mask=batch["all_atom_mask"],
            resolution=batch["resolution"],
            cutoff=self.plddt_loss_config.cutoff,
            num_bins=self.plddt_loss_config.num_bins,
            min_resolution=self.plddt_loss_config.min_resolution,
            max_resolution=self.plddt_loss_config.max_resolution,
            eps=self.plddt_loss_config.eps,
        )

        losses["experimentally_resolved"] = experimentally_resolved_loss(
            logits=outputs["experimentally_resolved_logits"],
            atom37_atom_exists=batch["atom37_atom_exists"],
            all_atom_mask=batch["all_atom_mask"],
            resolution=batch["resolution"],
            min_resolution=self.experimentally_resolved_loss_config.min_resolution,
            max_resolution=self.experimentally_resolved_loss_config.max_resolution,
            eps=self.experimentally_resolved_loss_config.eps,
        )

        losses["violation"] = violation_loss(
            violations=outputs["violations"],
            atom14_atom_exists=batch["atom14_atom_exists"],
            eps=self.violation_loss_config.eps,
        )

        if self.tm_loss_config.enabled:
            losses["tm"] = tm_loss(
                logits=outputs["tm_logits"],
                final_affine_tensor=outputs["final_affine_tensor"],
                backbone_rigid_tensor=batch["backbone_rigid_tensor"],
                backbone_rigid_mask=batch["backbone_rigid_mask"],
                resolution=batch["resolution"],
                max_bin=self.tm_loss_config.max_bin,
                num_bins=self.tm_loss_config.num_bins,
                min_resolution=self.tm_loss_config.min_resolution,
                max_resolution=self.tm_loss_config.max_resolution,
                eps=self.tm_loss_config.eps,
            )

        for loss in losses.values():
            assert loss.size() == (batch_size,)

        weighted_losses = {}
        weighted_losses["fape"] = losses["fape"] * self.fape_loss_config.weight
        weighted_losses["supervised_chi"] = losses["supervised_chi"] * self.supervised_chi_loss_config.weight
        weighted_losses["distogram"] = losses["distogram"] * self.distogram_loss_config.weight
        weighted_losses["masked_msa"] = losses["masked_msa"] * self.masked_msa_loss_config.weight
        weighted_losses["plddt_loss"] = losses["plddt_loss"] * self.plddt_loss_config.weight
        weighted_losses["experimentally_resolved"] = losses["experimentally_resolved"] * self.experimentally_resolved_loss_config.weight
        weighted_losses["violation"] = losses["violation"] * self.violation_loss_config.weight
        if self.tm_loss_config.enabled:
            weighted_losses["tm"] = losses["tm"] * self.tm_loss_config.weight

        for name in list(weighted_losses.keys()):
            loss = weighted_losses[name]
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.warning(f"Loss warning! weighted_losses['{name}']: {loss}")
                loss = torch.zeros_like(loss, requires_grad=True)
                weighted_losses[name] = loss

        weighted_total_loss = sum(weighted_losses.values())  # Not torch.sum

        # To decrease the relative importance of short sequences, we multiply the final loss
        # of each training example by the square root of the number of residues after cropping.
        assert batch["seq_length"].size() == (batch_size,)
        seq_length = batch["seq_length"].float()
        crop_size = torch.ones_like(seq_length) * batch["aatype"].size(1)
        scale = torch.sqrt(torch.minimum(seq_length, crop_size))
        scaled_weighted_total_loss = scale * weighted_total_loss

        losses = {key: tensor.detach().clone().mean() for key, tensor in losses.items()}
        losses["weighted_total_loss"] = weighted_total_loss.detach().clone().mean()
        losses["scaled_weighted_total_loss"] = scaled_weighted_total_loss.detach().clone().mean()

        scaled_weighted_total_loss = scaled_weighted_total_loss.mean()

        return scaled_weighted_total_loss, losses
