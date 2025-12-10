"""Base configuration dataclass for system configs."""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    """Configuration for a RAG system variant.

    Each configuration represents a different combination of
    chunking, retrieval, and generation strategies for ablation testing.

    Attributes:
        name: Short identifier for this config (e.g., "c0", "c1").
        description: Human-readable description of this config.
        use_smart_chunking: Use endpoint-centric chunking with $ref resolution.
        use_hybrid_search: Use BM25 + vector search with RRF fusion.
        use_metadata_filtering: Use query intent detection for API filtering.
        use_reranking: Use LLM-based reranking of top-K results.
        use_unknown_detection: Enhanced prompting for unknown detection.
    """

    name: str
    description: str
    use_smart_chunking: bool = False
    use_hybrid_search: bool = False
    use_metadata_filtering: bool = False
    use_reranking: bool = False
    use_unknown_detection: bool = False


# Registry of all configs by name
CONFIGS: dict[str, SystemConfig] = {}


def register_config(config: SystemConfig) -> SystemConfig:
    """Register a config in the global registry."""
    CONFIGS[config.name] = config
    return config


def get_config(name: str) -> SystemConfig:
    """Get a config by name from the registry."""
    if name not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(f"Unknown config: {name}. Available: {available}")
    return CONFIGS[name]
