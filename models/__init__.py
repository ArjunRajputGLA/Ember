"""
models package for EMBER 2018 Multimodal Malware Detection.
"""

from .cmanet import (
    CrossModalAttentionNet,
    CMANet_NoByte,
    CMANet_NoStruct,
    CMANet_NoText,
    CMANet_NoCrossAttn,
    MultimodalFusionNet,
    compute_model_capacity,
    verify_all_attention_safety_and_regressions,
    verify_ablation_isolation
)

__all__ = [
    "CrossModalAttentionNet",
    "CMANet_NoByte",
    "CMANet_NoStruct",
    "CMANet_NoText",
    "CMANet_NoCrossAttn",
    "MultimodalFusionNet",
    "compute_model_capacity",
    "verify_all_attention_safety_and_regressions",
    "verify_ablation_isolation"
]

