.PHONY: install dev test lint typecheck coverage clean run docker-up docker-down

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -v --tb=short

test-unit:
	pytest tests/unit -v --tb=short

test-integration:
	pytest tests/integration -v --tb=short

coverage:
	pytest --cov=src --cov-report=html --cov-report=term -v
	@echo "Coverage report: htmlcov/index.html"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
