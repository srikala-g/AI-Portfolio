.PHONY: help install dev lint format test run clean

help:            ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:         ## Install runtime package
	pip install -e .

dev:             ## Install dev tooling and pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

lint:            ## Run the linter
	ruff check .

format:          ## Auto-format the code
	ruff format .

test:            ## Run tests
	pytest -q

run:             ## Launch the Gradio app
	python app.py

clean:           ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info build dist
