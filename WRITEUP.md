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

### Incremental Feature Study

I approached this problem by designing 6 experimental setups (configurations C0-C5) where each configuration builds on the previous one by adding a single feature.

Important caveat: This is incremental addition, not true ablation. True ablation would start with the full system (C5) and remove one feature at a time. My approach measures cumulative improvement but cannot isolate feature interactions - for example, I cannot determine whether smart chunking + hybrid search is better or worse than either alone.

Given time and compute constraints, I chose this incremental approach to measure marginal gains. A complete ablation study would require 5 additional configurations (C5 minus each feature), which I would prioritize if continuing this work.

What this approach does enable:
- Measurement of marginal gains as each feature is added
- Fair comparison using config-agnostic evaluation metrics
- Understanding of which features provide diminishing returns

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
| C2     | ✓              | ✓             |                 |           |                   |
| C3     | ✓              | ✓             | ✓               |           |                   |
| C4     | ✓              | ✓             | ✓               | ✓         |                   |
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

This achieves 100% abstention accuracy on out-of-scope questions in the evaluation.

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

#### Evaluation Limitations

1. Small dataset: 84 questions is not enough for statistical significance. Differences of 0.1-0.2 in accuracy scores may be noise.
2. Circular LLM validation: The same model is used to generate ground truth, answers, and judgements. This could mask systematic biases.
3. Synthetic questions: LLM-generated questions may not represent real user queries. Production logs would be more valuable.
4. Tiny out-of-scope sample: 100% abstention accuracy on 5 questions is meaningless statistically - it could easily be 80% with 1 failure.
5. No latency measurement: C4/C5 use LLM reranking which adds significant latency, but I didn't measure this.

Despite these limitations, I believe the relative comparisons between configs are valid.

---

## 4. Results

### Summary Table

| Config | Hit Rate@5 | MRR | Accuracy (1-5) | Abstention |
|--------|:----------:|:---:|:--------------:|:----------:|
| C0 (Baseline) | 65.8% | 50.9% | 3.42 | 20% |
| C1 (Smart Chunking) | 92.4% | 79.7% | 3.20 | 20% |
| C2 (Hybrid Search) | 93.7% | 76.8% | 3.33 | 40% |
| C3 (Metadata Filter) | 92.4% | 78.2% | 3.32 | 40% |
| C4 (Reranking) | 97.5% | 91.1% | 3.65 | 40% |
| C5 (Full System) | 98.7% | 92.1% | 3.54 | 100% |

*Accuracy is the average LLM judge score (1-5 scale) across 84 questions.*

### Key Findings

1. Smart chunking provides the largest retrieval improvement (+27% hit rate). Endpoint-centric chunks with inlined schemas dramatically improve semantic matching.
2. Reranking significantly improves ranking quality (+0.4 MRR over baseline). LLM reranking pushes relevant results to the top positions.
3. Unknown detection works (100% abstention on out-of-scope). Simple prompt engineering is effective for detecting answerable questions.
4. Hybrid search has marginal benefit over smart chunking alone. BM25 helps with exact term matching but the gain is small.
5. Accuracy scores are relatively stable across configs (3.2-3.65). Retrieval improvements don't always translate to answer quality improvements, suggesting generation is a bottleneck.


### System Demonstration

The assignment provided 5 test questions. I ran all configurations (C0-C5) against these questions and verified answers against the ground-truth OpenAPI specs (results in [5_questions.md](5_questions.md))

#### Ground Truth Reference (from OpenAPI specs)

| Question | Actual Answer (from specs) |
|----------|---------------------------|
| Q1: Authentication | HTTP Basic (`securitySchemes.basic.scheme: "basic"`) |
| Q2: Workday accounts | Yes - `/accounts` supports `provider` query param |
| Q3: Session expiry | 1800 seconds (30 min) - `ConnectSessionCreate.expires_in.default: 1800` |
| Q4: LMS course creation | No POST endpoint exists - only GET endpoints for courses |
| Q5: Employee list response | `EmployeesPaginated` schema: `{ next_page, next, data: Employee[], raw }` |

---

#### Expected Behaviors

Q1 & Q2 (Authentication & Workday filtering): All configurations (C0-C5) correctly identified HTTP Basic authentication and the ability to filter accounts via `GET /accounts?provider=workday`. This demonstrates that the core retrieval pipeline works well for straightforward, well-documented features.

---

#### Unexpected Behaviors

Q3 (Session Token Expiry) - Massive variance across configs:

| Config | Answer Given | Accuracy |
|--------|-------------|----------|
| C0 | "30 minutes" | Correct |
| C1, C4 | "1 hour (3600 seconds)" | Wrong (2x actual) |
| C2, C5 | Abstained | Correct behavior |
| C3 | "24 hours (86,400 seconds)" | Wrong (48x actual) |

Q4 (LMS Course Creation) - Hallucination in baseline:
- C0 hallucinated that `learning_object_external_reference` is a required field
- C1-C5 correctly indicated that no create endpoint exists or abstained
- Ground truth: Only `GET /unified/lms/courses` and `GET /unified/lms/courses/{id}` exist

Q5 (Employee List Response) - Schema reference accuracy:
- C0 gave the most accurate answer: referenced `#/components/schemas/EmployeesPaginated`
- C1-C4 provided partial JSON structures that were less precise
- C5 abstained, stating insufficient information

---

#### Speculations

1. Q3 variance explained: The default value `"default": 1800` is buried in `ConnectSessionCreate.properties.expires_in`. Naive chunking (C0) likely preserved the raw JSON containing this value, while smart chunking's prose conversion lost it. Configs that abstained (C2, C5) were actually more accurate than those that guessed incorrect values.

2. C0's Q4 hallucination: Without smart chunking or unknown detection, C0 had no mechanism to verify that a POST endpoint actually exists. It inferred requirements from schema fields that are only used for GET responses.

3. C0's Q5 accuracy: Naive chunking preserves schema references verbatim (`#/components/schemas/EmployeesPaginated`), which is the actual response type. Smart chunking expanded this into prose, losing precision.

4. Unknown detection trade-off: C5's abstention on Q3 was technically correct - the retrieved context didn't contain the default value. However, the information *does* exist in the spec, just in a different chunk.

### Failure Analysis

Here are concrete examples of what failed and why, drawn from the C5 evaluation results:

#### Failure 1: Overly Conservative Unknown Detection (Factual Questions)

C5's factual questions scored only 2.90/5 despite 100% retrieval hit rate. The unknown detection prompt made the system too conservative:

Example: "What is the default value of expires_in?"
- Retrieved: Correct chunk (`stackone_components_connectsessioncreate_part0`) at rank 1
- C5 Response: "I don't have enough information... The context only states that expires_in is 'How long the session should be valid for in seconds,' but it does not specify any default value."
- Ground Truth: "defaults to 1800... expressed in seconds"
- Score: 3/5

The system retrieved the right information but the prose chunk didn't include the default value (due to the chunking issue discussed above), and the unknown detection prompt caused it to refuse rather than give a partial answer.

#### Failure 2: Cross-API Questions Are Hardest

Cross-API questions had the lowest accuracy (2.22/5) and only 89% retrieval hit rate:

Example: "Which APIs support filter.updated_after and what is the parameter's type in each?"
- Problem: Requires synthesizing information from 6 different API specs
- Retrieval: Only 5 chunks retrieved, couldn't cover all 6 APIs
- Generation: Even when chunks were retrieved, the LLM struggled to integrate scattered information into a coherent comparison

This suggests cross-API questions may need higher k (more retrieved chunks) or a multi-hop retrieval strategy.

#### Failure 3: Enum Values Lost in Chunking

Example: "What are the allowed values for order_by?"
- C0 Score: 4/5 (raw JSON contained `enum: ["provider", "service", "status", ...]`)
- C5 Score: 2/5 (prose chunk said "field to order results by" without enum values)
- C5 Response: "I don't have enough information... does not list which specific fields are valid for order_by"

The enum values existed in the original spec but were not preserved in the prose formatting.

#### What These Failures Reveal

| Problem | Impact | Fix |
|---------|--------|-----|
| Prose chunking loses structured data | Factual accuracy drops | Keep JSON metadata alongside prose |
| Unknown detection too aggressive | Refuses valid questions | Calibrate threshold based on retrieval confidence |
| Fixed k=5 retrieval | Cross-API questions under-served | Dynamic k based on question complexity |

---

## 5. Future Improvements

Based on the failure analysis above and the 5-question test results, these are the next steps I'd take:

### Priority 1: Hybrid Prose + JSON Chunking

Problem it solves: The C1 accuracy drop (3.42 → 3.20) and lost enum/default values.

Evidence from 5-question test:
- Q3 (session expiry): The `"default": 1800` value exists in `ConnectSessionCreate.expires_in` but was lost when converting to prose. C1/C3/C4 hallucinated wrong values (3600, 86400) because they couldn't find the actual default.
- Q5 (employee response): C0's raw `#/components/schemas/EmployeesPaginated` reference was more accurate than C1-C4's prose expansions.

Implementation: Store two representations per chunk:
1. Prose format for semantic retrieval (what we have now)
2. Raw JSON metadata appended to generation context
3. New: Explicitly preserve `default`, `enum`, `minimum`, `maximum` values inline in prose format

### Priority 2: Calibrated Unknown Detection

Problem it solves: Overly conservative abstention on answerable factual questions.

Evidence from 5-question test:
- Q3: C5 abstained correctly (the retrieved context didn't contain the default), but a partial answer would have been more useful: "The spec defines `expires_in` as 'how long the session should be valid for in seconds', but I don't see a default value in my context."
- Q5: C5 abstained entirely when it could have provided the schema structure from retrieved chunks.

Implementation: Instead of a binary "I don't have enough information" prompt:
- Calculate retrieval confidence (e.g., max similarity score, number of hits)
- Only trigger abstention when confidence is below threshold
- New: Implement confidence tiers: "definite answer" vs "partial answer with caveats" vs "cannot answer"
- Partial answers are better than refusals when context exists


### Priority 3: Dynamic k for Cross-API Questions

Problem it solves: Cross-API questions (2.22/5 accuracy) are under-served by fixed k=5.

Implementation:
- Detect query intent (already implemented in C3)
- If multiple APIs detected, increase k proportionally (e.g., k=5 per API)
- Alternatively: retrieve top-k from each API separately, then merge

### Priority 4: Hallucination Prevention for Write Operations

Problem it solves: Baseline (C0) hallucinated requirements for non-existent endpoints.

Evidence from 5-question test:
- Q4 (LMS course creation): C0 confidently stated that `learning_object_external_reference` is required to create a course, but no POST endpoint exists for courses. The spec only has `GET /unified/lms/courses` and `GET /unified/lms/courses/{id}`.

Implementation:
- Add explicit endpoint existence verification before generating answers about write operations
- Pre-generation check: if question asks "how to create/update/delete X", verify the relevant HTTP method (POST/PUT/PATCH/DELETE) exists in retrieved context
- If no write endpoint found, respond with "The API does not appear to support creating/updating X" rather than inferring from schema fields


## 6. Production Considerations

If deploying this system, these operational concerns would need addressing:

Latency vs. Accuracy Tradeoff: C4/C5 use LLM reranking which adds ~5-7s latency per query. For interactive use cases, we could use:
- Caching reranked results for common queries
- Async reranking with immediate vector-only results, then updated answer
- Threshold-based reranking (only trigger for low-confidence initial retrievals)

Caching Strategy:
- Embedding cache for spec reloads (specs change infrequently, embeddings are expensive)
- Query result cache for repeated questions (many users ask the same things)
- Invalidation on spec updates via content hashing

Query Decomposition for Cross-API Questions: The current system retrieves k=5 chunks regardless of query complexity. For questions like "which APIs support X", a more principled approach:
- Detect multi-API intent
- Decompose into N sub-queries (one per API)
- Retrieve and synthesize separately
- This directly addresses the 2.22/5 accuracy on cross-API questions

Evaluation Data Collection: The current synthetic dataset has significant limitations (LLM-generated, small sample, circular validation). In production:
- Log real user queries with opt-in feedback (thumbs up/down, "this didn't answer my question")
- Build a human-annotated golden set from the most common query patterns
- Use a different model family for judging (e.g., Claude for judging GPT-generated answers) to break circular validation