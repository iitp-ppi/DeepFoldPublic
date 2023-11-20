# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited

from typing import Optional


import torch
import torch.nn as nn

from deepfold.distributed.legacy import gather, scatter
from deepfold.model.alphafold.loss import compute_plddt, compute_predicted_aligned_error, _calculate_bin_centers
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear
from deepfold.utils.precision import is_fp16_enabled


def compute_tm(
    logits: torch.Tensor,  # [*, N, N, num_bins]
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    num_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])  # [N]
    assert residue_weights.ndim == 1

    boundaries = torch.linspace(0, max_bin, steps=(num_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)  #  [*, N, N, num_bins]

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)  # [*, N, N]

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())  # [N]
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)  # [*, N]

    weighted = per_alignment * residue_weights  # [*, N]

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(outputs["single"])
        aux_out["experimentally_resolved_logits"] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out.update(compute_predicted_aligned_error(tm_logits, **self.config.tm))

        return aux_out


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, num_bins, c_in, c_hidden):
        super().__init__()

        self.num_bins = num_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.num_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, num_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            num_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_z = c_z
        self.num_bins = num_bins

        self.linear = Linear(self.c_z, self.num_bins, init="final")

    def _forward(self, z):  # [*, N', N, C_z]
        """
        Args:
            z:
                [*, N', N, C_z] pair embedding
        Returns:
            [*, N', N, num_bins] distogram probability distribution
        """
        # [*, N', N, num_bins]
        logits = self.linear(z)
        logits = gather(logits, -3)
        logits = logits + logits.transpose(-2, -3)
        logits = scatter(logits, -3)
        return logits

    def forward(self, z):
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self._forward(z.float())
        else:
            return self._forward(z)


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, num_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            num_bins:
                Number of bins
        """
        super().__init__()

        self.c_z = c_z
        self.num_bins = num_bins

        self.linear = Linear(self.c_z, self.num_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, num_bins] prediction
        """
        # [*, N, N, num_bins]
        logits = self.linear(z)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(self, c_m, c_out, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, init="final")

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        # [*, N_seq, N_res, C_out]
        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super().__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits
