"""C3 Metadata Filtering configuration.

This config uses:
- All 7 OpenAPI specs (data foundation)
- Endpoint-centric chunking with $ref resolution (from C1)
- BM25 + Vector search with RRF fusion (from C2)
- Query intent detection for API filtering (NEW)
- No reranking
- No unknown detection prompting

This tests the impact of metadata filtering on retrieval precision.
"""

from ai_exercise.configs.base import SystemConfig, register_config

C3_METADATA_FILTERING = register_config(
    SystemConfig(
        name="c3",
        description="Metadata Filtering: Query intent detection for API filtering",
        use_smart_chunking=True,
        use_hybrid_search=True,
        use_metadata_filtering=True,
        use_reranking=False,
        use_unknown_detection=False,
    )
)
