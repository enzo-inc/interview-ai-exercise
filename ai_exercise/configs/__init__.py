"""Configuration module for system configs."""

from ai_exercise.configs.base import SystemConfig
from ai_exercise.configs.c0_baseline import C0_BASELINE
from ai_exercise.configs.c1_smart_chunking import C1_SMART_CHUNKING
from ai_exercise.configs.c2_hybrid import C2_HYBRID
from ai_exercise.configs.c3_metadata_filtering import C3_METADATA_FILTERING
from ai_exercise.configs.c4_reranking import C4_RERANKING

__all__ = [
    "C0_BASELINE",
    "C1_SMART_CHUNKING",
    "C2_HYBRID",
    "C3_METADATA_FILTERING",
    "C4_RERANKING",
    "SystemConfig",
]
