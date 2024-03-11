# Copyright 2024 DeepFold Team


"""DeepFold2 model configuration."""


from dataclasses import dataclass, field
from typing import Optional

NUM_RES = "num residues"
NUM_MSA_SEQ = "msa"
NUM_EXTRA_SEQ = "extra msa"
NUM_TEMPLATES = "num templates"


@dataclass
class InputEmbedderConfig:
    tf_dim: int = 22
    msa_dim: int = 49
    c_z: int = 128
    c_m: int = 256
    relpos_k: int = 32


@dataclass
class RecyclingEmbedderConfig:
    c_m: int = 256
    c_z: int = 128
    min_bin: float = 3.25
    max_bin: float = 20.75
    num_bins: int = 15
    inf: float = 1e8


@dataclass
class TemplateAngleEmbedderConfig:
    ta_dim: int = 57
    c_m: int = 256


@dataclass
class TemplatePairEmbedderConfig:
    tp_dim: int = 88
    c_t: int = 64


@dataclass
class TemplatePairStackConfig:
    c_t: int = 64
    c_hidden_tri_att: int = 16
    c_hidden_tri_mul: int = 64
    num_blocks: int = 2
    num_heads_tri: int = 4
    pair_transition_n: int = 2
    dropout_rate: float = 0.25
    inf: float = 1e9
    chunk_size_tri_att: Optional[int] = None


@dataclass
class TemplatePointwiseAttentionConfig:
    c_t: int = 64
    c_z: int = 128
    c_hidden: int = 16
    num_heads: int = 4
    inf: float = 1e5
    chunk_size: Optional[int] = None


@dataclass
class ExtraMSAEmbedderConfig:
    emsa_dim: int = 25
    c_e: int = 64


@dataclass
class ExtraMSAStackConfig:
    c_e: int = 64
    c_z: int = 128
    c_hidden_msa_att: int = 8
    c_hidden_opm: int = 32
    c_hidden_tri_mul: int = 128
    c_hidden_tri_att: int = 32
    num_heads_msa: int = 8
    num_heads_tri: int = 4
    num_blocks: int = 4
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps: float = 1e-8
    eps_opm: float = 1e-3
    chunk_size_msa_att: Optional[int] = None
    chunk_size_opm: Optional[int] = None
    chunk_size_tri_att: Optional[int] = None


@dataclass
class EvoformerStackConfig:
    c_m: int = 256
    c_z: int = 128
    c_hidden_msa_att: int = 32
    c_hidden_opm: int = 32
    c_hidden_tri_mul: int = 128
    c_hidden_tri_att: int = 32
    c_s: int = 384
    num_heads_msa: int = 8
    num_heads_tri: int = 4
    num_blocks: int = 48
    transition_n: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps_opm: float = 1e-3
    chunk_size_msa_att: Optional[int] = None
    chunk_size_opm: Optional[int] = None
    chunk_size_tri_att: Optional[int] = None


@dataclass
class StructureModuleConfig:
    c_s: int = 384
    c_z: int = 128
    c_hidden_ipa: int = 16
    c_hidden_ang_res: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_ang_res_blocks: int = 2
    num_angles: int = 7
    scale_factor: float = 10.0
    inf: float = 1e5
    eps: float = 1e-8


@dataclass
class PerResidueLDDTCaPredictorConfig:
    c_s: int = 384
    c_hidden: int = 128
    num_bins: int = 50


@dataclass
class DistogramHeadConfig:
    c_z: int = 128
    num_bins: int = 64


@dataclass
class MaskedMSAHeadConfig:
    c_m: int = 256
    c_out: int = 23


@dataclass
class ExperimentallyResolvedHeadConfig:
    c_s: int = 384
    c_out: int = 37


@dataclass
class TMScoreHeadConfig:
    c_z: int = 128
    num_bins: int = 64
    max_bin: int = 31


@dataclass
class AuxiliaryHeadsConfig:
    per_residue_lddt_ca_predictor_config: PerResidueLDDTCaPredictorConfig = field(
        default=PerResidueLDDTCaPredictorConfig(),
    )
    distogram_head_config: DistogramHeadConfig = field(
        default=DistogramHeadConfig(),
    )
    masked_msa_head_config: MaskedMSAHeadConfig = field(
        default=MaskedMSAHeadConfig(),
    )
    experimentally_resolved_head_config: ExperimentallyResolvedHeadConfig = field(
        default=ExperimentallyResolvedHeadConfig(),
    )
    tm_score_head_config: TMScoreHeadConfig = field(
        default=TMScoreHeadConfig(),
    )
    tm_score_head_enabled: bool = False


@dataclass
class FAPELossConfig:
    weight: float = 1.0
    backbone_clamp_distance: float = 10.0
    backbone_loss_unit_distance: float = 10.0
    backbone_weight: float = 0.5
    sidechain_clamp_distance: float = 10.0
    sidechain_length_scale: float = 10.0
    sidechain_weight: float = 0.5
    eps: float = 1e-4


@dataclass
class SupervisedChiLossConfig:
    weight: float = 1.0
    chi_weight: float = 0.5
    angle_norm_weight: float = 0.01
    eps: float = 1e-8


@dataclass
class DistogramLossConfig:
    weight: float = 0.3
    min_bin: float = 2.3125
    max_bin: float = 21.6875
    num_bins: int = 64
    eps: float = 1e-8


@dataclass
class MaskedMSALossConfig:
    weight: float = 2.0
    eps: float = 1e-8


@dataclass
class PLDDTLossConfig:
    weight: float = 0.01
    cutoff: float = 15.0
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    num_bins: int = 50
    eps: float = 1e-8


@dataclass
class ExperimentallyResolvedLossConfig:
    weight: float = 0.0
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    eps: float = 1e-8


@dataclass
class ViolationLossConfig:
    weight: float = 0.0
    violation_tolerance_factor: float = 12.0
    clash_overlap_tolerance: float = 1.5
    eps: float = 1e-8


@dataclass
class TMLossConfig:
    enabled: bool = False
    weight: float = 0.0
    min_resolution: float = 0.1
    max_resolution: float = 3.0
    num_bins: int = 64
    max_bin: int = 31
    eps: float = 1e-8


@dataclass
class LossConfig:
    fape_loss_config: FAPELossConfig = field(
        default=FAPELossConfig(),
    )
    supervised_chi_loss_config: SupervisedChiLossConfig = field(
        default=SupervisedChiLossConfig(),
    )
    distogram_loss_config: DistogramLossConfig = field(
        default=DistogramLossConfig(),
    )
    masked_msa_loss_config: MaskedMSALossConfig = field(
        default=MaskedMSALossConfig(),
    )
    plddt_loss_config: PLDDTLossConfig = field(
        default=PLDDTLossConfig(),
    )
    experimentally_resolved_loss_config: ExperimentallyResolvedLossConfig = field(
        default=ExperimentallyResolvedLossConfig(),
    )
    violation_loss_config: ViolationLossConfig = field(
        default=ViolationLossConfig(),
    )
    tm_loss_config: TMLossConfig = field(
        default=TMLossConfig(),
    )
