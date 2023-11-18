# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


import torch
import torch.nn as nn

from deepfold.distributed.legacy import gather, scatter
from deepfold.model.alphafold.loss import compute_plddt, compute_predicted_aligned_error, compute_tm
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear
from deepfold.utils.precision import is_fp16_enabled


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
            aux_out["predicted_tm_score"] = compute_tm(tm_logits, **self.config.tm)
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
