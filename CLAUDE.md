# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stellabook is an AI-powered service that retrieves arXiv research papers and generates interactive Jupyter notebooks for educational purposes. It uses multiple LLM providers for different tasks (paper analysis and notebook generation), orchestrated through LangChain.

## Commands

All commands use `uv` as the package manager and `just` as the task runner. See the `justfile` for available commands. Run `just` with no arguments to list them.

## Architecture

### Notebook Generation Pipeline

```
POST /generate (arxiv_id, interactive)
  → arxiv_client: fetch paper metadata (Atom XML)
  → figure_extractor: download PDF, extract/compress images
  → generator: analyze paper and generate structured notebook content via LLMs
  → notebook_builder.build_notebook(): assemble .ipynb with figures, imports
  → return notebook file
```

### Module Responsibilities

- **`config.py`** — Central configuration: LLM provider setup, arXiv constraints, figure limits
- **`arxiv_client.py`** — Async arXiv API client with rate limiting and secure XML parsing
- **`generator.py`** — LLM-driven steps in the notebook generation pipeline, each using a different provider. Contains the system prompts
- **`figure_extractor.py`** — Downloads paper PDFs and extracts/compresses embedded images
- **`notebook_builder.py`** — Assembles the final Jupyter notebook from generated content, figures, and metadata
- **`models.py`** / **`notebook_models.py`** — Pydantic models for arXiv data and notebook structure
- **`fastapi_app.py`** — FastAPI server exposing the notebook generation pipeline as a REST API

### Key Design Decisions

- **Multiple LLM providers**: Different models are used for different steps in the notebook generation pipeline. All accessed via LangChain abstractions in `config.py`.
- **Structured output**: `generate_notebook_content()` uses LangChain's `.with_structured_output()` to return a typed `NotebookContent` Pydantic model.
- **Figure handling**: Images under 40KB are inlined as base64 HTML; larger ones use notebook attachments. Images are progressively scaled down to meet size constraints.
- **Import extraction**: `notebook_builder.extract_imports()` parses code cells with `ast` and maps module names to PyPI packages (e.g., `cv2` → `opencv-python`).

## Testing

Tests use `pytest` with `pytest-asyncio` (auto mode) and `pytest-httpx`. HTTP calls are mocked with `httpx.MockTransport`. LLM calls are mocked with `unittest.mock`. Sample Atom XML feeds are defined in `conftest.py`.

## Environment Variables

Requires `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` (see `.env.example`).

## Code Standards

- Python 3.12+ with strict Pyright type checking
- Ruff for linting (rules: E, F, I, UP) and formatting
- All public functions have type annotations
- Async/await throughout the HTTP and notebook generation pipeline