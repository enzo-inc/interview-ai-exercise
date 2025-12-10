"""C4 Re-ranking configuration.

This config uses:
- All 7 OpenAPI specs (data foundation)
- Endpoint-centric chunking with $ref resolution (from C1)
- BM25 + Vector search with RRF fusion (from C2)
- Query intent detection for API filtering (from C3)
- LLM-based re-ranking of top-K results (NEW)
- No unknown detection prompting

This tests the impact of re-ranking on retrieval precision.
"""

from ai_exercise.configs.base import SystemConfig, register_config

C4_RERANKING = register_config(
    SystemConfig(
        name="c4",
        description="Re-ranking: LLM-based re-ranking of top-K results",
        use_smart_chunking=True,
        use_hybrid_search=True,
        use_metadata_filtering=True,
        use_reranking=True,
        use_unknown_detection=False,
    )
)
