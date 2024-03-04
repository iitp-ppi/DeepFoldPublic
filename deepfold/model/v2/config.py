# Copyright 2024 DeepFold Team


"""DeepFold2 model configuration."""


import logging
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


PAIR_REPRESENTATION_DIM = 128
MSA_REPRESENTATION_DIM = 256
SINGLE_REPRESENTATION_DIM = 384
EXTRA_MSA_REPRESENTATION_DIM = 64
TEMPLATE_REPRESENTATION_DIM = 64


@dataclass(kw_only=True)
class _ConfigBase:
    r"""
    Base class for configurations.
    """

    def to_omegaconf(self):
        return OmegaConf.structured(self)

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        for k, v in self.__annotations__.items():
            if not issubclass(v, _ConfigBase):
                continue
            com = getattr(self, k)
            if com is None:
                setattr(self, k, v())
            elif isinstance(com, (dict, DictConfig)):
                setattr(self, k, v(**com))


@dataclass(kw_only=True)
class EmbedderConfig(_ConfigBase):

    # Input and recylcing embedder
    input_embedder: "InputEmbedderConfig" = None
    recycling_embedder: "RecyclingEmbedderConfig" = None
    # Template embedders
    template_angle_embedder: "TemplateAngleEmbedderConfig" = None
    template_pair_embedder: "TemplatePairEmbedderConfig" = None
    template_pair_stack: "TemplatePairStackConfig" = None
    template_projector: "TemplateProjectorConfig" = None
    # Extra MSA embedders
    extra_msa_embedder: "ExtraMsaEmbedderConfig" = None
    extra_msa_stack: "ExtraMsaStackConfig" = None


@dataclass(kw_only=True)
class InputEmbedderConfig(_ConfigBase):
    tf_dim: int = 22  # 21
    msa_dim: int = 49
    c_z: int = PAIR_REPRESENTATION_DIM
    c_m: int = MSA_REPRESENTATION_DIM
    max_relative_index: int = 32  # relpos_k
    use_chain_relative: bool = False  # True
    max_relative_chain: int = 2


@dataclass(kw_only=True)
class RecyclingEmbedderConfig(_ConfigBase):
    c_z: int = PAIR_REPRESENTATION_DIM
    c_m: int = MSA_REPRESENTATION_DIM
    recycle_positions: bool = True
    min_bin: float = 3.25
    max_bin: float = 20.75
    num_bins: int = 15
    inf: float = 1e8


@dataclass(kw_only=True)
class TemplateAngleEmbedderConfig(_ConfigBase):
    ta_dim: int = 57  # 34
    c_m: int = MSA_REPRESENTATION_DIM


@dataclass(kw_only=True)
class TemplatePairEmbedderConfig(_ConfigBase):
    c_z: int = PAIR_REPRESENTATION_DIM
    tp_dim: list[int] = [88]  # [39, 1, 22, 22, 1, 1, 1, 1]
    c_t: int = TEMPLATE_REPRESENTATION_DIM
    v2_feature: bool = False


@dataclass(kw_only=True)
class TemplatePairStackConfig(_ConfigBase):
    c_t: int = TEMPLATE_REPRESENTATION_DIM
    c_hidden_tri_att: int = 16
    c_hidden_tri_mul: int = 64
    num_blocks: int = 2
    num_heads_tri: int = 4
    num_pair_transitions: int = 2
    chunk_size: int = 128
    chunk_size_tri_att: Optional[int] = None
    dropout_rate: float = 0.25
    inf: float = 1e9
    tri_att_first: bool = True


@dataclass(kw_only=True)
class TemplateProjectorConfig(_ConfigBase):
    c_t: int = TEMPLATE_REPRESENTATION_DIM
    c_z: int = PAIR_REPRESENTATION_DIM
    enable_point_att: bool = True
    c_hidden: int = 16
    num_heads: int = 4
    chunk_size: Optional[int] = None
    inf: float = 1e5


@dataclass(kw_only=True)
class ExtraMsaEmbedderConfig(_ConfigBase):
    extra_msa_dim: int = 25
    c_e: int = EXTRA_MSA_REPRESENTATION_DIM


@dataclass(kw_only=True)
class ExtraMsaStackConfig(_ConfigBase):
    c_e: int = EXTRA_MSA_REPRESENTATION_DIM
    c_z: int = PAIR_REPRESENTATION_DIM
    c_hidden_msa_att: int = 8
    c_hidden_opm: int = 32
    c_hidden_tri_mul: int = 128
    c_hidden_tri_att: int = 32
    num_heads_msa: int = 8
    num_heads_tri: int = 4
    num_blocks: int = 4
    num_msa_transitions: int = 4
    num_pair_transitions: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps: float = 1e-8
    eps_opm: float = 1e-3
    chunk_size_msa_att: Optional[int] = None
    chunk_size_opm: Optional[int] = None
    chunk_size_tri_att: Optional[int] = None


@dataclass(kw_only=True)
class EvoformerStackConfig(_ConfigBase):
    c_m: int = MSA_REPRESENTATION_DIM
    c_z: int = PAIR_REPRESENTATION_DIM
    c_hidden_msa_att: int = 32
    c_hidden_opm: int = 32
    c_hidden_tri_mul: int = 128
    c_hidden_tri_att: int = 32
    c_s: int = SINGLE_REPRESENTATION_DIM
    num_heads_msa: int = 8
    num_heads_tri: int = 4
    num_blocks: int = 48
    num_msa_transitions: int = 4
    num_pair_transitions: int = 4
    msa_dropout: float = 0.15
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps_opm: float = 1e-3
    chunk_size_msa_att: Optional[int] = None
    chunk_size_opm: Optional[int] = None
    chunk_size_tri_att: Optional[int] = None


@dataclass(kw_only=True)
class HeadsConfig(_ConfigBase):
    distogram: "DistogramHeadConfig" = None
    experimentally_resolved: "ExperimentallyResolvedHeadConfig" = None
    masked_msa: "MaskedMsaHeadConfig" = None
    predicted_aligned_error: "PredictedAlignedErrorHeadConfig" = None
    predicted_lddt: "PredictedPlddtHeadConfig" = None
    structure_module: "StructureModuleConfig" = None


@dataclass(kw_only=True)
class StructureModuleConfig(_ConfigBase):
    c_s: int = SINGLE_REPRESENTATION_DIM
    c_z: int = PAIR_REPRESENTATION_DIM

    c_hidden_ipa: int = 16
    c_hidden_ang_res: int = 128
    num_heads_ipa: int = 12
    num_qk_points: int = 4
    num_v_points: int = 8
    dropout_rate: float = 0.1
    num_blocks: int = 8
    num_ang_res_blocks: int = 2
    num_angles: int = 7
    scale_factor: float = 10.0  # AA to nm
    inf: float = 1e5
    eps: float = 1e-8  # TODO: Check


@dataclass(kw_only=True)
class DistogramHeadConfig(_ConfigBase):
    c_z: int = PAIR_REPRESENTATION_DIM
    output_dim: int = 64


@dataclass(kw_only=True)
class ExperimentallyResolvedHeadConfig(_ConfigBase):
    enabled: bool = False
    c_s: int = SINGLE_REPRESENTATION_DIM
    output_dim: int = 37


@dataclass(kw_only=True)
class MaskedMsaHeadConfig(_ConfigBase):
    c_m: int = MSA_REPRESENTATION_DIM
    output_dim: int = 23  # 22 for multimer


@dataclass(kw_only=True)
class PredictedAlignedErrorHeadConfig(_ConfigBase):
    enabled: bool = False
    c_z: int = PAIR_REPRESENTATION_DIM
    output_dim: int = 64
    iptm_weight: float = 0.8


@dataclass(kw_only=True)
class PredictedPlddtHeadConfig(_ConfigBase):
    c_s: int = SINGLE_REPRESENTATION_DIM
    output_dim: int = 50
    hidden_dim: int = 128


@dataclass(kw_only=True)
class DeepFoldLossConfig(_ConfigBase):
    # distogram
    # experimentally_resolved
    # fape
    # plddt
    # masked_msa
    # supervised_chi
    # violation
    # predicted_aligned_error
    # chain_center_of_mass
    pass


@dataclass(kw_only=True)
class DeepFoldConfig(_ConfigBase):
    r"""
    This is the configuration class to store the configuration of a DeepFold model. It is used to instantiate a
    DeepFold model according to the specified arguments, defining model architecture.
    """

    multimer: bool = False
    precision: str = "fp32"

    # Modules configuratiaon
    embedder: "EmbedderConfig" = field(default_factory=EmbedderConfig())
    encoder: "EvoformerStackConfig" = field(default_factory=EvoformerStackConfig())
    decoder: "HeadsConfig" = field(default_factory=HeadsConfig())

    # Training loss configuration
    loss_config: DeepFoldLossConfig = field(default=DeepFoldLossConfig())

    # Recycling (last dimension in the batch dict):
    num_recycling_iters: int = 3  # 20 for multimer

    # Fused Adam + SWA
    fused_adam_swa: bool = True

    # Triton MHA
    triton_mha: bool = False

    # Evoformer Attention
    evo_attn: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.triton_mha and self.evo_attn:
            raise ValueError("Triton multi-head attention and Evoformer attention cannot be enabled simultaneously")

        if self.precision in {"fp32", "tf32", "bf16"}:
            pass
        elif self.precision in {"amp", "fp16"}:
            self.embedder.recycling_embedder.inf = 1e4
            self.embedder.template_pair_stack.inf = 1e4
            self.embedder.template_projector.inf = 1e4
            self.embedder.extra_msa_stack.inf = 1e4
            self.encoder.inf = 1e4
            self.template_pair_feat_inf = 1e4
        else:
            raise ValueError(f"Unknown precision '{repr(self.precision)}'")
