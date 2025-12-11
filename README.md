# StackOne OpenAPI RAG System

RAG system for answering questions about StackOne's 7 OpenAPI specifications with ablation study evaluation.

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- OpenAI API key

### Setup

1. Clone and enter the repository
2. Copy `.env_example` to `.env` and add your `OPENAI_API_KEY`
3. Run the setup script:
   ```bash
   python scripts/setup.py
   ```
   This installs dependencies and creates indices for all configurations (C0-C5).

### Running the Demo

```bash
make start-app
```

Access at http://localhost:8501

### Running Evaluations

```bash
# Evaluate a specific config
make eval CONFIG=c0

# Compare multiple configs
make eval-compare CONFIGS=c0,c1,c2,c3,c4,c5

# View results comparison in Streamlit
make compare-evals
```

Results are saved to `reports/results/`.

### Configurations

| Config | Description |
|--------|-------------|
| c0 | Baseline: naive chunking, vector search |
| c1 | Smart endpoint-centric chunking |
| c2 | Hybrid search (BM25 + vector with RRF) |
| c3 | Query intent detection for API filtering |
| c4 | LLM-based reranking |
| c5 | Full system with unknown detection |

See [WRITEUP.md](WRITEUP.md) for detailed explanation of the approach and results.

---

# Original README

## AI Exercise - Retrieval

> simple RAG example

## Project requirements

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to install and manage python dependencies.

### Docker Engine (optional)

Install [Docker Engine](https://docs.docker.com/engine/install/) to build and run the API's Docker image locally.

## Installation

```bash
make install
```

## API

The project includes an API built with [FastAPI](https://fastapi.tiangolo.com/). Its code can be found at `src/api`.

The API is containerized using a [Docker](https://docs.docker.com/get-started/) image, built from the `Dockerfile` and `docker-compose.yml` at the root. This is optional, you can also run the API without docker.

### Environment Variables

Copy .env_example to .env and fill in the values.

### Build and start the API

To build and start the API, use the following Makefile command:

```bash
make dev-api
```

you can also use `make start-api` to start the API using Docker.

## Frontend

The project includes a frontend built with [Streamlit](https://streamlit.io/). Its code can be found at `demo`.

Run the frontend with:

```bash
make start-app
```

## Testing

To run unit tests, run `pytest` with:

```bash
make test
```

## Formatting and static analysis

There is some preset up formatting and static analysis tools to help you write clean code. check the make file for more details.

```bash
make lint
```

```bash
make format
```

```bash
make typecheck
```

# Get Started

Have a look in `ai_exercise/constants.py`. Then check out the server routes in `ai_exercise/main.py`. 

1. Load some documents by calling the `/load` endpoint. Does the system work as intended? Are there any issues?

2. Find some method of evaluating the quality of the retrieval system.

3. See how you can improve the retrieval system. Some ideas:
- Play with the chunking logic
- Try different embeddings models
- Other types of models which may be relevant
- How else could you store the data for better retrieval?
