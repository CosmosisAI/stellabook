"""Central configuration: LLM providers, arXiv constraints, OTEL tracing."""

import os

from dotenv import load_dotenv

load_dotenv()

# LangChain OTEL tracing — set before any langchain import so that
# LangChain picks up the OpenTelemetry exporter on first use.
os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
os.environ.setdefault("LANGSMITH_TRACING", "true")

from langchain_anthropic import ChatAnthropic  # noqa: E402
from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: E402

ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.0
ARXIV_DEFAULT_MAX_RESULTS = 10
ARXIV_MAX_RESULTS_LIMIT = 100

RESEARCH_MODEL_NAME = "gemini-3.1-flash-lite-preview"
RESEARCH_MAX_TOKENS = 4096
NOTEBOOK_MODEL_NAME = "claude-sonnet-4-6"
NOTEBOOK_MAX_TOKENS = 16384

NOTEBOOK_NBFORMAT_VERSION = 4
NOTEBOOK_NBFORMAT_MINOR = 5

FIGURE_MIN_DIMENSION = 100
FIGURE_MAX_COUNT = 20
FIGURE_INLINE_MAX_BYTES = 40_960
FIGURE_SCALED_MIN_DIMENSION = 600


def get_research_model() -> BaseChatModel:
    """Return the LLM used for paper research analysis."""
    return ChatGoogleGenerativeAI(  # type: ignore[call-arg]
        model=RESEARCH_MODEL_NAME,
        max_output_tokens=RESEARCH_MAX_TOKENS,
    )


def get_notebook_model() -> BaseChatModel:
    """Return the LLM used for structured notebook generation."""
    return ChatAnthropic(
        model_name=NOTEBOOK_MODEL_NAME,  # type: ignore[call-arg]
        max_tokens=NOTEBOOK_MAX_TOKENS,  # type: ignore[call-arg]
    )
