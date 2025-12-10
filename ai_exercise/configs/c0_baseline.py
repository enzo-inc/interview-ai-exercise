"""C0 Baseline configuration.

This config uses:
- All 7 OpenAPI specs (data foundation)
- Naive chunking (current implementation)
- Pure vector search (no hybrid)
- No metadata filtering
- No reranking
- No unknown detection prompting

This establishes the baseline for ablation studies.
"""

from ai_exercise.configs.base import SystemConfig, register_config

C0_BASELINE = register_config(
    SystemConfig(
        name="c0",
        description="Baseline: All 7 specs, naive chunking, pure vector search",
        use_smart_chunking=False,
        use_hybrid_search=False,
        use_metadata_filtering=False,
        use_reranking=False,
        use_unknown_detection=False,
    )
)
