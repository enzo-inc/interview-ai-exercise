"""Configuration module for system configs."""

from ai_exercise.configs.base import SystemConfig
from ai_exercise.configs.c0_baseline import C0_BASELINE
from ai_exercise.configs.c1_smart_chunking import C1_SMART_CHUNKING

__all__ = ["C0_BASELINE", "C1_SMART_CHUNKING", "SystemConfig"]
