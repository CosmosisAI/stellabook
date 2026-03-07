# Stellabook Commands

default:
    @just --list

## Setup / Environment
install:
    uv sync

## Server
serve:
    uv run stellabook-api

## Testing
test *args:
    uv run python -m pytest {{args}}

## Linting & Formatting
lint:
    uv run python -m ruff check src tests

fix:
    uv run python -m ruff check --fix src tests

format:
    uv run python -m ruff format src tests

format-check:
    uv run python -m ruff format --check src tests

check: lint typecheck

## Type Checking
typecheck:
    uv run python -m pyright src

## Cleanup
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .ruff_cache .pyright .pytest_cache
