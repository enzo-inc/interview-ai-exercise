# StackOne RAG System - Technical Writeup

## 1. Problem Statement

This project addresses the StackOne AI Engineer assignment: building a RAG (Retrieval-Augmented Generation) system that can answer questions about StackOne's 7 OpenAPI specifications:

- stackone - Core platform API
- hris - Human Resources Information System
- ats - Applicant Tracking System
- lms - Learning Management System
- iam - Identity & Access Management
- crm - Customer Relationship Management
- marketing - Marketing automation

### Requirements

1. Load all 7 OpenAPI specs into a vector store
2. Answer questions accurately using retrieved context
3. Gracefully indicate limitations when unable to answer
4. Build evaluation methods for the retrieval system
5. Suggest and implement improvements

---

## 2. Approach & Architecture

### Ablation Study Design

I approached this problem by designing 6 experimental setups (configurations C0-C5) where each configuration introduces an additional feature to independently assess its effect on retrieval and answer performance.

This approach enables:
- Quantitative measurement of each improvement's contribution
- Fair comparison using config-agnostic evaluation metrics
- Clear understanding of which techniques provide the most value

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   OpenAPI    │    │   Chunking   │    │   Indexing   │              │
│  │   Specs (7)  │───▶│   Strategy   │───▶│  ChromaDB +  │              │
│  │              │    │              │    │    BM25      │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                 │                        │
│                                                 ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   Answer     │◀───│  Generation  │◀───│  Retrieval   │              │
│  │   + Sources  │    │   (LLM)      │    │  Pipeline    │              │
│  └──────────────┘    └──────────────┘    └──────────────┘              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Matrix

| Config | Smart Chunking | Hybrid Search | Metadata Filter | Reranking | Unknown Detection |
|--------|:--------------:|:-------------:|:---------------:|:---------:|:-----------------:|
| C0     |                |               |                 |           |                   |
| C1     | ✓              |               |                 |           |                   |
| C2     |                | ✓             |                 |           |                   |
| C3     |                |               | ✓               |           |                   |
| C4     |                |               |                 | ✓         |                   |
| C5     | ✓              | ✓             | ✓               | ✓         | ✓                 |

---

## 3. Implementation Details

### 3.1 Document Loading & Chunking

#### Naive Chunking (C0)

The baseline approach splits OpenAPI specs by top-level JSON keys:
- Each path becomes a chunk
- Each schema becomes a chunk
- Each webhook becomes a chunk

Limitations: Loses context when related information spans multiple keys. A POST endpoint's request body schema is stored separately from the endpoint itself.

#### Smart Chunking (C1+)

Endpoint-centric chunking creates semantically coherent chunks:

1. $ref Resolution: Inlines referenced schemas up to depth 2, so endpoint chunks contain their request/response schemas
2. Prose Formatting: Converts JSON to human-readable documentation format
3. Standalone Schema Chunks: Creates separate chunks for reusable schemas

Example smart chunk:
```
POST /unified/hris/employees
Creates a new employee record.

Request Body (HrisCreateEmployeeRequestDto):
- first_name (string, required): Employee's first name
- last_name (string, required): Employee's last name
- email (string): Employee's email address
...
```

#### Structural IDs

To enable fair evaluation across different chunking strategies, each chunk is tagged with structural IDs indicating what OpenAPI content it covers:

| Content Type | Format | Example |
|--------------|--------|---------|
| Endpoint | `{api}.paths.{path}.{method}` | `hris.paths./unified/hris/employees.post` |
| Schema | `{api}.components.{name}` | `hris.components.Employee` |

This allows retrieval metrics to be computed consistently regardless of how chunks are split.

### 3.2 Retrieval Strategies

#### Pure Vector Search (C0, C1)

- Embeddings: OpenAI `text-embedding-3-small` (1536 dimensions)
- Vector Store: ChromaDB with cosine similarity
- Top-K: Returns 5 most similar chunks

#### BM25 Lexical Search

Keyword-based search using the `rank-bm25` library. Particularly effective for:
- Exact technical terms (endpoint paths, field names)
- Queries with specific identifiers

#### Hybrid Search (C2)

Combines vector and BM25 using Reciprocal Rank Fusion (RRF):

```python
rrf_score = alpha / (k + rank_vector) + (1 - alpha) / (k + rank_bm25)
```

Where `k=60` (standard RRF constant) and `alpha=0.5` (equal weighting).

#### Query Intent Detection (C3)

LLM-based classifier determines which APIs are relevant to a query:

```
Query: "How do I create an employee?"
→ Detected APIs: ["hris"]
→ Filter retrieval to HRIS chunks only
```

Reduces noise from irrelevant APIs and improves precision.

#### LLM Reranking (C4)

Two-stage retrieval:
1. Retrieve 3× candidates (15 chunks)
2. LLM reranks to select top-K (5 chunks)

The reranker considers semantic relevance beyond embedding similarity.

### 3.3 Unknown Detection

C5 implements graceful limitation handling through prompt engineering:

```
System prompt addition:
"If the provided context does not contain sufficient information to answer
the question, respond with: 'I don't have enough information in the StackOne
documentation to answer this question.'"
```

This achieves 100% abstention accuracy on out-of-scope questions in our evaluation.

### 3.4 Evaluation Framework

#### Synthetic Dataset Generation

Since manual question creation is time-consuming, I used LLM-assisted generation:

1. Sample random paths/schemas from each API (context limits prevent using full specs)
2. Generate questions across 6 categories:
   - Factual: General facts about APIs
   - Endpoint: How to use specific endpoints
   - Schema: Data model questions
   - Auth: Authentication questions
   - Cross-API: Questions spanning multiple APIs
   - Out-of-Scope: Questions the system should refuse

3. Extract ground truth answers and relevant structural IDs

Dataset size: 84 questions across all categories.

#### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| Hit Rate@K | % of queries with at least one relevant chunk in top-K |
| MRR | Mean Reciprocal Rank - average of 1/rank for first relevant result |

Relevance is determined by structural ID matching: if a retrieved chunk's `covers` metadata includes any of the question's `relevant_structural_ids`, it's a hit.

#### Answer Quality Metrics

| Metric | Description |
|--------|-------------|
| Accuracy Score | LLM judge rating (1-5 scale) comparing answer to ground truth |
| Abstention Accuracy | % of out-of-scope questions correctly refused |

The LLM judge evaluates whether the generated answer contains the same factual information as the ground truth, regardless of phrasing.

---

## 4. Results

### Summary Table

| Config | Hit Rate@5 | MRR (%) | Accuracy (%) | Abstention (%) |
|--------|:----------:|:-------:|:------------:|:--------------:|
| C0 (Baseline) | 65.8% | 50.9% | 68.4% | 20% |
| C1 (Smart Chunking) | 92.4% | 79.7% | 64.0% | 20% |
| C2 (Hybrid Search) | 93.7% | 76.8% | 66.6% | 40% |
| C3 (Metadata Filter) | 92.4% | 78.2% | 66.4% | 40% |
| C4 (Reranking) | 97.5% | 91.1% | 73.0% | 40% |
| C5 (Full System) | 98.7% | 92.1% | 70.8% | 100% |

### Key Findings

1. Smart chunking provides the largest retrieval improvement (+27% hit rate). Endpoint-centric chunks with inlined schemas dramatically improve semantic matching.

2. Reranking significantly improves ranking quality (+0.4 MRR over baseline). LLM reranking pushes relevant results to the top positions.

3. Unknown detection works (100% abstention on out-of-scope). Simple prompt engineering is effective for detecting answerable questions.

4. Hybrid search has marginal benefit over smart chunking alone. BM25 helps with exact term matching but the gain is small.

5. Accuracy scores are relatively stable across configs (3.2-3.65). Retrieval improvements don't always translate to answer quality improvements, suggesting generation is a bottleneck.

---

## 5. Future Improvements

### 5.1 Retrieval Enhancements

#### Query Expansion & Rewriting

Current queries are used as-is. Improvements:
- HyDE (Hypothetical Document Embeddings): Generate a hypothetical answer, embed that instead of the query
- Query decomposition: Break complex queries into sub-queries
- Synonym expansion: Add API-specific synonyms (e.g., "employee" → "worker", "staff")

#### Multi-Vector Retrieval

Current single-vector approach has limitations. Alternatives:
- ColBERT: Late interaction model that computes similarity at the token level, better for exact matching
- Multi-vector embeddings: Store multiple vectors per chunk (e.g., title + content separately)

#### Learned Sparse Representations

- SPLADE: Learned sparse vectors that combine semantic understanding with keyword matching
- Better than manual hybrid search as weights are learned

#### Hierarchical Retrieval

Current flat retrieval doesn't leverage document structure:
- Coarse-to-fine: First retrieve relevant APIs, then relevant endpoints
- Parent-child chunks: Small chunks for retrieval, larger chunks for context

#### Contextual Retrieval

Anthropic's contextual retrieval approach:
- Prepend each chunk with LLM-generated context about where it fits in the overall document
- Improves retrieval for chunks that lack standalone context

### 5.2 Embedding Model Improvements

#### Domain-Specific Fine-Tuning

Current `text-embedding-3-small` is general-purpose. Options:
- Fine-tune on OpenAPI documentation
- Use contrastive learning with question-chunk pairs from our evaluation set

#### Larger Context Embeddings

Current 8K token limit can truncate large chunks:
- jina-embeddings-v3: 8K context, multiple task-specific adapters
- Voyage AI: Up to 16K context embeddings

#### Matryoshka Embeddings

For production efficiency:
- Use Matryoshka Representation Learning (MRL) models
- Allows using smaller embedding dimensions without retraining
- Trade-off between accuracy and storage/speed

### 5.3 Generation Improvements

#### Chain-of-Thought Reasoning

Current generation is single-shot. Improvements:
- Have the model reason about which retrieved chunks are relevant
- Step-by-step answer construction

#### Self-Consistency / Verification

- Generate multiple answers, select most consistent
- Post-generation fact-checking against retrieved context
- Citation extraction to verify claims are grounded

#### Structured Output

For API documentation questions:
- Generate structured responses (JSON with endpoint, parameters, etc.)
- Better for integration with developer tools

### 5.4 Index & Storage Improvements

#### Knowledge Graph Integration

OpenAPI specs have rich structure:
- Build a knowledge graph of APIs, endpoints, schemas, relationships
- Combine graph traversal with vector search
- Better for multi-hop questions ("What schema does the employee endpoint return?")

#### ANN Optimizations

Current ChromaDB with default HNSW settings:
- Tune HNSW parameters (ef_construction, M) for better recall/speed trade-off
- Consider IVF indexes for larger scale

#### Quantization

For production scale:
- Product quantization (PQ) for memory efficiency
- Binary quantization for speed
- Trade-off: ~5% recall loss for 32x memory reduction

### 5.5 Caching & Performance

#### Query Result Caching

- Cache frequent query results (many API questions are repeated)
- Semantic cache: similar queries return cached results
- TTL-based invalidation when specs update

#### Embedding Caching

- Cache embeddings for repeated document loads
- Useful for development iteration

#### Batch Processing

- Current evaluation runs queries sequentially
- Batch embedding and generation calls for throughput

### 5.6 Evaluation Improvements

#### Human-in-the-Loop Validation

Current ground truth is LLM-generated:
- Validate a sample manually
- Identify systematic errors in generation

#### Separate Model Families

Using same model (GPT-4) for generation and judging may introduce bias:
- Use Claude for judging GPT-4 outputs (or vice versa)
- Multiple judges with voting

#### Larger Test Sets

Current 84 questions is small:
- Generate more questions per category
- Add edge cases (ambiguous queries, multi-part questions)
- Include real user questions from production logs

#### End-to-End User Testing

- A/B test configurations with real users
- Measure task completion, not just accuracy scores
- Track follow-up question rates (lower is better)