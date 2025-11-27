.PHONY: help install install-dev setup clean run test format lint

help:
	@echo "Available commands:"
	@echo "  make setup       - Install uv and create virtual environment"
	@echo "  make install     - Install project dependencies"
	@echo "  make install-dev - Install project with dev dependencies"
	@echo "  make run         - Run the main.py example"
	@echo "  make clean       - Remove virtual environment and cache files"
	@echo "  make format      - Format code with black"
	@echo "  make lint        - Lint code with ruff"

setup:
	@echo "🚀 Setting up project with uv..."
	@command -v uv >/dev/null 2>&1 || (echo "Installing uv..." && curl -LsSf https://astral.sh/uv/install.sh | sh)
	uv venv
	@echo "✅ Setup complete! Activate with: source .venv/bin/activate"

install:
	@echo "📥 Installing dependencies..."
	uv pip install -e .
	@echo "✅ Dependencies installed!"

install-dev:
	@echo "📥 Installing dependencies with dev tools..."
	uv pip install -e ".[dev]"
	@echo "✅ All dependencies installed!"

run:
	@echo "🏃 Running main.py..."
	python main.py

clean:
	@echo "🧹 Cleaning up..."
	rm -rf .venv
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"

format:
	@echo "🎨 Formatting code..."
	black .
	@echo "✅ Code formatted!"

lint:
	@echo "🔍 Linting code..."
	ruff check .
	@echo "✅ Linting complete!"

test:
	@echo "🧪 Running tests..."
	pytest
	@echo "✅ Tests complete!"

