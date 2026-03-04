# Contributing to ML Monitoring Dashboard

## Development Setup

```bash
git clone <repo-url>
cd ml-monitoring-dashboard
make dev
```

## Workflow

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Run the full test suite: `make test`
4. Ensure lint passes: `make lint`
5. Ensure type checks pass: `make typecheck`
6. Submit a pull request

## Code Style

- Python 3.11+ features are welcome
- Use `ruff` for formatting and linting
- Use type annotations throughout
- Follow existing patterns in the codebase

## Testing

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Use `pytest-asyncio` for async tests
- Use SQLite in-memory databases for test isolation

## Commit Messages

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation
- `test:` test additions/changes
- `refactor:` code refactoring
