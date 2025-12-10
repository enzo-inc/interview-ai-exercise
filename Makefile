install:
	uv sync --all-extras

########################################################################################################################
# Quality checks
########################################################################################################################

test:
	uv run pytest tests

lint:
	uv run ruff check ai_exercise tests

format:
	uv run ruff check ai_exercise tests --fix

typecheck:
	uv run mypy ai_exercise


########################################################################################################################
# Api
########################################################################################################################

start-api:
	docker compose up

dev-api:
	uv run ai_exercise/main.py

########################################################################################################################
# Streamlit
########################################################################################################################

start-app:
	uv run streamlit run demo/main.py

compare-evals:
	uv run streamlit run demo/compare.py

########################################################################################################################
# Evaluation
########################################################################################################################

# Run full evaluation for a config (e.g., make eval CONFIG=c0)
eval:
	uv run python -m ai_exercise.evals.runner run --config $(CONFIG)

# Run retrieval-only evaluation
eval-retrieval:
	uv run python -m ai_exercise.evals.runner run --config $(CONFIG) --type retrieval --no-judges

# Run end-to-end evaluation with LLM judges
eval-e2e:
	uv run python -m ai_exercise.evals.runner run --config $(CONFIG) --type e2e

# Compare two or more configs (e.g., make eval-compare CONFIGS=c0,c1)
eval-compare:
	uv run python -m ai_exercise.evals.runner compare --configs $(CONFIGS)

# Generate markdown report from all results
eval-report:
	uv run python -m ai_exercise.evals.runner report --output reports/

# Generate synthetic evaluation dataset
generate-eval-questions:
	uv run python -m ai_exercise.evals.generate_dataset

# Load data into vector store for a specific config (e.g., make load-data CONFIG=c0)
load-data:
	uv run python -m ai_exercise.loading.loader $(CONFIG)
