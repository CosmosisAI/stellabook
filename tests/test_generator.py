"""Tests for the generator module."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from stellabook.generator import (
    _build_notebook_user_message,
    _build_user_message,
    generate_notebook_content,
    research_paper,
)
from stellabook.models import Author, Category, Paper
from stellabook.notebook_models import CellType, Figure


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

        mock_message = AsyncMock()
        mock_message.content = [AsyncMock(text=SAMPLE_RESEARCH)]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_message

        result = await research_paper(paper, client=mock_client)

        assert isinstance(result, str)
        assert "Addresses gap in X." in result
        assert "Insight 1" in result
        mock_client.messages.create.assert_called_once()

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert "Test Paper Title" in call_kwargs["messages"][0]["content"]


def _make_tool_use_block(
    tool_input: dict[str, object],
) -> AsyncMock:
    """Create a mock tool_use content block."""
    block = AsyncMock()
    block.type = "tool_use"
    block.name = "create_notebook"
    block.input = tool_input
    return block


class TestGenerateNotebookContent:
    async def test_calls_model_with_tool_use(self) -> None:
        paper = _make_paper()
        tool_input = {
            "title": "Generated Notebook",
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": "# Intro",
                },
                {
                    "cell_type": "code",
                    "source": "import numpy as np",
                },
            ],
        }

        mock_message = AsyncMock()
        mock_message.stop_reason = "tool_use"
        mock_message.content = [_make_tool_use_block(tool_input)]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_message

        result = await generate_notebook_content(
            paper, SAMPLE_RESEARCH, client=mock_client
        )

        assert result.title == "Generated Notebook"
        assert len(result.cells) == 2
        assert result.cells[0].cell_type == CellType.MARKDOWN
        assert result.cells[1].cell_type == CellType.CODE
        mock_client.messages.create.assert_called_once()

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert call_kwargs["tools"] is not None
        assert call_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "create_notebook",
        }
        user_msg = call_kwargs["messages"][0]["content"]
        assert "Test Paper Title" in user_msg
        assert "Research Analysis:" in user_msg
        assert "Addresses gap in X." in user_msg

    async def test_raises_on_max_tokens(self) -> None:
        paper = _make_paper()

        mock_message = AsyncMock()
        mock_message.stop_reason = "max_tokens"
        mock_message.content = []

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="truncated"):
            await generate_notebook_content(paper, SAMPLE_RESEARCH, client=mock_client)

    async def test_raises_when_no_tool_call(self) -> None:
        paper = _make_paper()

        # Model returns a text block instead of tool use
        text_block = AsyncMock()
        text_block.type = "text"

        mock_message = AsyncMock()
        mock_message.stop_reason = "end_turn"
        mock_message.content = [text_block]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_message

        with pytest.raises(ValueError, match="did not call the create_notebook tool"):
            await generate_notebook_content(paper, SAMPLE_RESEARCH, client=mock_client)

    async def test_handles_latex_in_tool_input(self) -> None:
        """Tool use handles LaTeX escaping correctly."""
        paper = _make_paper()
        tool_input = {
            "title": "Math Notebook",
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ("The formula is $\\alpha + \\beta = \\gamma$"),
                },
            ],
        }

        mock_message = AsyncMock()
        mock_message.stop_reason = "tool_use"
        mock_message.content = [_make_tool_use_block(tool_input)]

        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_message

        result = await generate_notebook_content(
            paper, SAMPLE_RESEARCH, client=mock_client
        )

        assert "\\alpha" in result.cells[0].source
        assert "\\beta" in result.cells[0].source
