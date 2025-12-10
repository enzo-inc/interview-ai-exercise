"""C1 Smart Chunking configuration.

This config uses:
- All 7 OpenAPI specs (data foundation)
- Endpoint-centric chunking with $ref resolution (NEW)
- Pure vector search (no hybrid)
- No metadata filtering
- No reranking
- No unknown detection prompting

This tests the impact of semantic chunking on retrieval quality.
"""

from ai_exercise.configs.base import SystemConfig, register_config

C1_SMART_CHUNKING = register_config(
    SystemConfig(
        name="c1",
        description="Smart Chunking: Endpoint-centric chunks with $ref resolution",
        use_smart_chunking=True,
        use_hybrid_search=False,
        use_metadata_filtering=False,
        use_reranking=False,
        use_unknown_detection=False,
    )
)
