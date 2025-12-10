"""C5 Full System configuration.

This config uses:
- All 7 OpenAPI specs (data foundation)
- Endpoint-centric chunking with $ref resolution (from C1)
- BM25 + Vector search with RRF fusion (from C2)
- Query intent detection for API filtering (from C3)
- LLM-based re-ranking of top-K results (from C4)
- Enhanced prompting for unknown detection (NEW)

This is the production-ready configuration with all improvements enabled.
"""

from ai_exercise.configs.base import SystemConfig, register_config

C5_FULL = register_config(
    SystemConfig(
        name="c5",
        description="Full System: All improvements + unknown detection",
        use_smart_chunking=True,
        use_hybrid_search=True,
        use_metadata_filtering=True,
        use_reranking=True,
        use_unknown_detection=True,
    )
)
