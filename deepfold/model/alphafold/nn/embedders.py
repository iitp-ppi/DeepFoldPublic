# Copyright 2023 DeepFold Team
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited


from typing import Tuple

import torch
import torch.nn as nn

from deepfold.distributed.legacy import gather
from deepfold.model.alphafold.nn.primitives import LayerNorm, Linear
from deepfold.utils.debug import dump_args
from deepfold.utils.tensor_utils import add, one_hot


class InputEmbedder(nn.Module):
    """
    Implements AF2 Alg. 3 InputEmbedder and 4 relpos.
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        **kwargs,
    ) -> None:
        """
        Args:
            tf_dim: int
                Final dimension of the target features.
            msa_dim: int
                Final dimension of the MSA features.
            c_z: int
                Pair embedding dimension
            c_m: int
                MSA embedding dimension
            relpos_k: int
                Window size used in relative positional encoding
        """
        super().__init__()

        # Initial embeddings
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # Relative postion encoding
        self.relpos_k = relpos_k
        self.num_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.num_bins, c_z)

    def relpos(self, ri: torch.Tensor) -> torch.Tensor:
        """
        Computes relative positional encodings

        Implements AF2 Alg. 4

        Args:
            ri: torch.Tensor
                "residue_index" features of shape [*, N]
        """
        d = ri[..., :, None] - ri[..., None, :]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        d = one_hot(d, boundaries).to(ri.dtype)
        return self.linear_relpos(d)

    @dump_args
    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf: torch.Tensor [*, N_res, tf_dim]
                "target_feat" features of shape
            ri: torch.Tensor [*, N_res]
                "residue_index" features of shape.
            msa: torch.Tensor [*, N_clust, N_res, msa_dim]
                "msa_feat" features of shape.
        Returns:
            msa_emb: torch.Tensor [*, N_clust, N_res, C_m]
                MSA embedding.
            pair_emb: torch.Tensor [*, N_res, N_res, C_z]
                Pair embedding.
        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = add(pair_emb, tf_emb_i[..., :, None, :], inplace=inplace_safe)
        pair_emb = add(pair_emb, tf_emb_j[..., None, :, :], inplace=inplace_safe)

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = self.linear_tf_m(tf).unsqueeze(-3).expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class ParallelInputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        relpos_k: int,
        **kwargs,
    ) -> None:
        super().__init__()

        # Initial embeddings
        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_m = c_m
        self.c_z = c_z

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        # Relative postion encoding
        self.relpos_k = relpos_k
        self.num_bins = 2 * relpos_k + 1
        self.linear_relpos = Linear(self.num_bins, c_z)

    def relpos(self, ri: torch.Tensor) -> torch.Tensor:
        ri_all = gather(ri, dim=-1)
        d = ri[..., :, None] - ri_all[..., None, :]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        d = one_hot(d, boundaries).to(ri.dtype)
        return self.linear_relpos(d)

    @dump_args
    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf: "target_feat" [*, N', tf_dim]
            ri: "residue_index" [*, N']
            msa: "msa_feat" [*, S, N', msa_dim]
        Returns:
            msa_emb: [*, S, N', C_m]
            pair_emb: [*, N', N, C_z]
        """
        # [*, N', C_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N', N, C_z]
        pair_emb = self.relpos(ri.type(tf_emb_i.dtype))
        pair_emb = pair_emb + tf_emb_i[..., :, None, :]

        tf_emb_j = gather(tf_emb_j, dim=-2)
        pair_emb = pair_emb + tf_emb_j[..., None, :, :]
        # TODO: Chunk

        # [*, S, N', C_m]
        n_clust = msa.shape[-3]
        tf_m = self.linear_tf_m(tf).unsqueeze(-3).expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        num_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            num_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins
        self.inf = inf

        self.linear = Linear(self.num_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    @dump_args
    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.num_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins**2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)
        d = torch.sum((x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True)

        # [*, N, N, num_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = z_update + d

        return m_update, z_update


class ParallelRecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """

    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        num_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            num_bins:
                Number of distogram bins
        """
        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins
        self.inf = inf

        self.linear = Linear(self.num_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    @dump_args
    def forward(
        self,
        m_1: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m_1:
                [*, N', C_m]
            z:
                [*, N', N, C_z]
            x:
                [*, N', 3]
        Returns:
            m_1:
                [*, N', C_m]
            z:
                [*, N', N, C_z]
        """
        # [*, N', C_m]
        m_update = self.layer_norm_m(m_1)

        # [*, N', N, C_z]
        z_update = self.layer_norm_z(z)

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.num_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins**2
        upper = torch.cat([squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1)

        x_all = gather(x, dim=-2)
        d = torch.sum((x[..., :, None, :] - x_all[..., None, :, :]) ** 2, dim=-1, keepdims=True)
        # Chunk

        # [*, N', N, num_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N', N, C_z]
        d = self.linear(d)
        z_update = z_update + d

        return m_update, z_update


class TemplateAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    @dump_args
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out)

    @dump_args
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)

        return x
