"""C2 Hybrid Retrieval configuration.

This config uses:
- All 7 OpenAPI specs (data foundation)
- Endpoint-centric chunking with $ref resolution (from C1)
- BM25 + Vector search with RRF fusion (NEW)
- No metadata filtering
- No reranking
- No unknown detection prompting

This tests the impact of hybrid retrieval on precision for keyword-heavy queries.
"""

from ai_exercise.configs.base import SystemConfig, register_config

C2_HYBRID = register_config(
    SystemConfig(
        name="c2",
        description="Hybrid Retrieval: BM25 + Vector with RRF fusion",
        use_smart_chunking=True,
        use_hybrid_search=True,
        use_metadata_filtering=False,
        use_reranking=False,
        use_unknown_detection=False,
    )
)
