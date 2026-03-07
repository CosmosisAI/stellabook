"""Tests for the generator module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from stellabook.generator import (
    _build_notebook_user_message,
    _build_user_message,
    generate_notebook_content,
    research_paper,
)
from stellabook.models import Author, Category, Paper
from stellabook.notebook_models import CellType, Figure, NotebookContent


def _make_paper() -> Paper:
    return Paper(
        arxiv_id="2301.07041",
        title="Test Paper Title",
        summary="This is a test abstract.",
        authors=[Author(name="Alice"), Author(name="Bob")],
        categories=[Category(term="cs.AI"), Category(term="cs.LG")],
        links=[],
        published=datetime(2023, 1, 17, tzinfo=UTC),
        updated=datetime(2023, 1, 17, tzinfo=UTC),
        primary_category="cs.AI",
    )


SAMPLE_RESEARCH = """\
## Background and Motivation
Addresses gap in X.

## Objectives
Prove Y.

## Key Insights
- Insight 1
- Insight 2

## Novel Approach
New method Z.

## Key Takeaways
- Takeaway 1
- Takeaway 2

## Practical Applications
Apply to domain D.

## Suggested Code Demonstrations
- Demo attention mechanism
"""


class TestBuildUserMessage:
    def test_contains_paper_fields(self) -> None:
        paper = _make_paper()
        msg = _build_user_message(paper)

        assert "Test Paper Title" in msg
        assert "Alice" in msg
        assert "Bob" in msg
        assert "cs.AI" in msg
        assert "2301.07041" in msg
        assert "This is a test abstract." in msg
        assert "https://arxiv.org/abs/2301.07041" in msg


class TestBuildNotebookUserMessage:
    def test_contains_research_fields(self) -> None:
        paper = _make_paper()
        msg = _build_notebook_user_message(paper, SAMPLE_RESEARCH)

        assert "Test Paper Title" in msg
        assert "Research Analysis:" in msg
        assert "Addresses gap in X." in msg
        assert "Prove Y." in msg
        assert "Insight 1" in msg
        assert "New method Z." in msg
        assert "Demo attention mechanism" in msg

    def test_includes_figures_section_when_provided(self) -> None:
        paper = _make_paper()
        figures = [
            Figure(
                label="figure_0",
                image_base64="AAAA",
                page_number=1,
                width=200,
                height=150,
            ),
            Figure(
                label="figure_1",
                image_base64="BBBB",
                page_number=3,
                width=400,
                height=300,
            ),
        ]
        msg = _build_notebook_user_message(paper, SAMPLE_RESEARCH, figures)

        assert "Available Figures:" in msg
        assert "figure_0: page 1, 200x150 pixels" in msg
        assert "figure_1: page 3, 400x300 pixels" in msg

    def test_no_figures_section_when_empty(self) -> None:
        paper = _make_paper()
        msg = _build_notebook_user_message(paper, SAMPLE_RESEARCH, [])

        assert "Available Figures:" not in msg

    def test_no_figures_section_when_none(self) -> None:
        paper = _make_paper()
        msg = _build_notebook_user_message(paper, SAMPLE_RESEARCH, None)

        assert "Available Figures:" not in msg


class TestResearchPaper:
    async def test_calls_model_and_returns_markdown(self) -> None:
        paper = _make_paper()

        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AIMessage(content=SAMPLE_RESEARCH)

        result = await research_paper(paper, model=mock_model)

        assert isinstance(result, str)
        assert "Addresses gap in X." in result
        assert "Insight 1" in result
        mock_model.ainvoke.assert_called_once()

        messages = mock_model.ainvoke.call_args.args[0]
        assert "Test Paper Title" in messages[1].content


class TestGenerateNotebookContent:
    async def test_calls_model_with_structured_output(self) -> None:
        paper = _make_paper()
        parsed = NotebookContent(
            title="Generated Notebook",
            cells=[
                {"cell_type": "markdown", "source": "# Intro"},  # type: ignore[list-item]
                {"cell_type": "code", "source": "import numpy as np"},  # type: ignore[list-item]
            ],
        )

        raw_message = AIMessage(content="")
        raw_message.response_metadata = {"stop_reason": "end_turn"}

        structured_model = AsyncMock()
        structured_model.ainvoke.return_value = {
            "raw": raw_message,
            "parsed": parsed,
            "parsing_error": None,
        }

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = structured_model

        result = await generate_notebook_content(
            paper, SAMPLE_RESEARCH, model=mock_model
        )

        assert result.title == "Generated Notebook"
        assert len(result.cells) == 2
        assert result.cells[0].cell_type == CellType.MARKDOWN
        assert result.cells[1].cell_type == CellType.CODE
        mock_model.with_structured_output.assert_called_once_with(
            NotebookContent, include_raw=True
        )
        structured_model.ainvoke.assert_called_once()

        messages = structured_model.ainvoke.call_args.args[0]
        assert "Test Paper Title" in messages[1].content
        assert "Research Analysis:" in messages[1].content
        assert "Addresses gap in X." in messages[1].content

    async def test_raises_on_max_tokens(self) -> None:
        paper = _make_paper()
        parsed = NotebookContent(
            title="Partial",
            cells=[{"cell_type": "markdown", "source": "# Partial"}],  # type: ignore[list-item]
        )

        raw_message = AIMessage(content="")
        raw_message.response_metadata = {"stop_reason": "max_tokens"}

        structured_model = AsyncMock()
        structured_model.ainvoke.return_value = {
            "raw": raw_message,
            "parsed": parsed,
            "parsing_error": None,
        }

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = structured_model

        with pytest.raises(ValueError, match="truncated"):
            await generate_notebook_content(paper, SAMPLE_RESEARCH, model=mock_model)

    async def test_raises_on_parsing_error(self) -> None:
        paper = _make_paper()

        raw_message = AIMessage(content="")
        raw_message.response_metadata = {"stop_reason": "end_turn"}

        structured_model = AsyncMock()
        structured_model.ainvoke.return_value = {
            "raw": raw_message,
            "parsed": None,
            "parsing_error": "Invalid JSON in model output",
        }

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = structured_model

        with pytest.raises(ValueError, match="Failed to parse notebook content"):
            await generate_notebook_content(paper, SAMPLE_RESEARCH, model=mock_model)

    async def test_handles_latex_in_structured_output(self) -> None:
        """Structured output handles LaTeX escaping correctly."""
        paper = _make_paper()
        parsed = NotebookContent(
            title="Math Notebook",
            cells=[
                {  # type: ignore[list-item]
                    "cell_type": "markdown",
                    "source": "The formula is $\\alpha + \\beta = \\gamma$",
                },
            ],
        )

        raw_message = AIMessage(content="")
        raw_message.response_metadata = {"stop_reason": "end_turn"}

        structured_model = AsyncMock()
        structured_model.ainvoke.return_value = {
            "raw": raw_message,
            "parsed": parsed,
            "parsing_error": None,
        }

        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = structured_model

        result = await generate_notebook_content(
            paper, SAMPLE_RESEARCH, model=mock_model
        )

        assert "\\alpha" in result.cells[0].source
        assert "\\beta" in result.cells[0].source
