.PHONY: help setup download-data download-weights format lint train evaluate clean

PYTHON := uv run python
DATA_URL := https://www.mimuw.edu.pl/~mwrochna/upload/synthetic_shapes.zip
SAM_URL := https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

help:
	@echo "Available commands:"
	@echo "  make setup          - Install project dependencies via uv"
	@echo "  make download-data  - Download and extract the dataset"
	@echo "  make download-weights - Download SAM checkpoint weights"
	@echo "  make format         - Auto-format code and fix linting errors"
	@echo "  make lint           - Check code for linting and formatting issues"
	@echo "  make train          - Train the ResNet18 classifier"
	@echo "  make evaluate       - Run the SAM pipeline with both approaches"
	@echo "  make clean          - Remove cache and temporary files"

setup:
	uv sync --all-extras

download-data:
	mkdir -p data/
	curl -L -A "Mozilla/5.0" -o data/synthetic_shapes.zip $(DATA_URL)
	unzip -o -q data/synthetic_shapes.zip -d data/
	rm data/synthetic_shapes.zip

download-weights:
	curl -L -o sam_vit_b_01ec64.pth $(SAM_URL)

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .

train:
	$(PYTHON) -m src.train_classifier --data_dir data/synthetic_shapes --epochs 5 --batch_size 128

evaluate:
	@echo "Running Approach 1 (Foreground Only)..."
	$(PYTHON) -m src.evaluate --approach 1
	@echo "Running Approach 2 (Foreground + Background)..."
	$(PYTHON) -m src.evaluate --approach 2

clean:
	rm -rf .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +