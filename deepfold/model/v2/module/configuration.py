# Copyright 2024 DeepFold Team


"""DeepFold2 model configuration."""


import logging
from dataclasses import dataclass

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
class DeepFoldConfig(_ConfigBase):
    r"""
    This is the configuration class to store the configuration of a DeepFold model. It is used to instantiate a
    DeepFold model according to the specified arguments, defining model architecture.
    """

    fp16: bool = False
    bf16: bool = False

    max_recycles: int = 3

    embedder: "EmbedderConfig" = None
    encoder: "EvoformerStackConfig" = None
    decoder: "DecoderConfig" = None

    def __post_init__(self):
        super().__post_init__()
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot be enabled simultaneously")


@dataclass(kw_only=True)
class EmbedderConfig(_ConfigBase):

    # Input and recylcing embedder
    input_embedder: "InputEmbedderConfig" = None
    recycling_embedder: "RecyclingEmbedderConfig" = None
    # Template embedders
    template_angle_embedder: "TemplateAngleEmbedderConfig" = None
    template_pair_embedder: "TemplatePairEmbedderConfig" = None
    template_pair_stack: "TemplatePairStackConfig" = None
    template_pointwise_attention: "TemplatePointwiseAttentionConfig" = None
    # Extra MSA embedders
    extra_msa_embedder: "ExtraMsaEmbedderConfig" = None
    extra_msa_stack: "ExtraMsaStackConfig" = None


@dataclass(kw_only=True)
class InputEmbedderConfig(_ConfigBase):
    target_feature_dim: int = 22  # 21
    msa_feature_dim: int = 49
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    msa_representation_dim: int = MSA_REPRESENTATION_DIM
    max_relative_idx: int = 32
    use_chain_relative: bool = False  # True
    max_relative_chain: int = 2


@dataclass(kw_only=True)
class RecyclingEmbedderConfig(_ConfigBase):
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    msa_representation_dim: int = MSA_REPRESENTATION_DIM
    recycle_positions: bool = True
    min_bin: float = 3.25
    max_bin: float = 20.75
    num_bins: int = 15
    inf: float = 1e8


@dataclass(kw_only=True)
class TemplateAngleEmbedderConfig(_ConfigBase):
    template_angle_feature_dim: int = 57
    msa_representation_dim: int = MSA_REPRESENTATION_DIM


@dataclass(kw_only=True)
class TemplatePairEmbedderConfig(_ConfigBase):
    template_pair_feature_dim: int = 88
    template_representation_dim: int = TEMPLATE_REPRESENTATION_DIM


@dataclass(kw_only=True)
class TemplatePairStackConfig(_ConfigBase):
    template_representation_dim: int = TEMPLATE_REPRESENTATION_DIM
    chunk_size: int = 128
    num_blocks: int = 2
    num_heads: int = 4
    num_transitions: int = 2
    dropout_rate: float = 0.25
    inf: float = 1e9
    tri_attn_first: bool = True


@dataclass(kw_only=True)
class TemplatePointwiseAttentionConfig(_ConfigBase):
    average_template: bool = False
    template_representation_dim: int = TEMPLATE_REPRESENTATION_DIM
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    hidden_dim: int = 16
    num_heads: int = 4
    inf: float = 1e5


@dataclass(kw_only=True)
class ExtraMsaEmbedderConfig(_ConfigBase):
    extra_msa_feature_dim: int = 25
    extra_msa_representation_dim: int = EXTRA_MSA_REPRESENTATION_DIM


@dataclass(kw_only=True)
class ExtraMsaStackConfig(_ConfigBase):
    extra_msa_representation_dim: int = EXTRA_MSA_REPRESENTATION_DIM
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    num_blocks: int = 4
    opm_hidden_dim: int = 32
    omp_first: bool = False
    msa_attn_hidden_dim: int = 8
    num_msa_attn_heads: int = 8
    num_msa_transitions: int = 4
    msa_dropout: float = 0.15
    tri_mul_hidden_dim: int = 128
    tri_attn_hidden_dim: int = 32
    num_tri_attn_heads: int = 4
    num_pair_transitions: int = 4
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps: float = 1e-10


@dataclass(kw_only=True)
class EvoformerStackConfig(_ConfigBase):
    msa_representation_dim: int = MSA_REPRESENTATION_DIM
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    num_blocks: int = 48
    opm_hidden_dim: int = 32
    omp_first: bool = False
    msa_attn_hidden_dim: int = 32
    num_msa_attn_heads: int = 8
    num_msa_transitions: int = 4
    msa_dropout: float = 0.15
    tri_mul_hidden_dim: int = 128
    tri_attn_hidden_dim: int = 32
    num_tri_attn_heads: int = 4
    num_pair_transitions: int = 4
    pair_dropout: float = 0.25
    inf: float = 1e9
    eps: float = 1e-10


@dataclass(kw_only=True)
class DecoderConfig(_ConfigBase):
    distogram: "DistogramHeadConfig" = None
    experimentally_resolved: "ExperimentallyResolvedHeadConfig" = None
    masked_msa: "MaskedMsaHeadConfig" = None
    predicted_aligned_error: "PredictedAlignedErrorHeadConfig" = None
    predicted_lddt: "PredictedPlddtHeadConfig" = None
    structure_module: "StructureModuleConfig" = None


@dataclass(kw_only=True)
class StructureModuleConfig(_ConfigBase):
    single_representation_dim: int = SINGLE_REPRESENTATION_DIM
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    angle_representation_dim: int = 128
    num_blocks: int = 8
    ipa_hidden_dim: int = 16
    num_ipa_heads: int = 12
    num_qk_points: int = 4
    num_v_porints: int = 8
    dropout_rate: float = 0.1
    # num_transition_layers: int = 3
    num_resnet_blocks: int = 2
    num_angles: int = 7
    scale_factor: float = 10.0  # AA to nm
    eps: float = 1e-12
    inf: float = 1e5


@dataclass(kw_only=True)
class DistogramHeadConfig(_ConfigBase):
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    output_dim: int = 64


@dataclass(kw_only=True)
class ExperimentallyResolvedHeadConfig(_ConfigBase):
    enabled: bool = False
    single_representation_dim: int = SINGLE_REPRESENTATION_DIM
    output_dim: int = 37


@dataclass(kw_only=True)
class MaskedMsaHeadConfig(_ConfigBase):
    msa_representation_dim: int = MSA_REPRESENTATION_DIM
    output_dim: int = 23  # 22 for multimer


@dataclass(kw_only=True)
class PredictedAlignedErrorHeadConfig(_ConfigBase):
    enabled: bool = False
    pair_representation_dim: int = PAIR_REPRESENTATION_DIM
    output_dim: int = 64
    iptm_weight: float = 0.8


@dataclass(kw_only=True)
class PredictedPlddtHeadConfig(_ConfigBase):
    single_representation_dim: int = SINGLE_REPRESENTATION_DIM
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
