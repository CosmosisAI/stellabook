"""Tests for the FastAPI application."""

import json
from collections.abc import Generator
from datetime import UTC, datetime
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from stellabook.fastapi_app import app
from stellabook.models import Author, Category, Paper, PipelineContext
from stellabook.notebook_models import CellType, NotebookCell, NotebookContent

_MOCK_PAPER = Paper(
    arxiv_id="2301.07041",
    title="Test Paper",
    summary="A test abstract.",
    authors=[Author(name="Alice"), Author(name="Bob")],
    categories=[Category(term="cs.AI")],
    links=[],
    published=datetime(2023, 1, 17, tzinfo=UTC),
    updated=datetime(2023, 1, 17, tzinfo=UTC),
    primary_category="cs.AI",
)


class PipelineMocks(NamedTuple):
    """Mocks for the notebook generation pipeline."""

    arxiv_client: MagicMock
    download_pdf: MagicMock
    extract_figures: MagicMock
    extract_text: MagicMock
    research_paper: MagicMock
    generate_content: MagicMock
    research_model: MagicMock
    notebook_model: MagicMock


@pytest.fixture
def pipeline_mocks() -> Generator[PipelineMocks, None, None]:
    mock_content = NotebookContent(
        title="Test Notebook",
        cells=[
            NotebookCell(cell_type=CellType.MARKDOWN, source="# Test"),
            NotebookCell(cell_type=CellType.CODE, source="x = 1"),
        ],
    )

    mock_research_model = MagicMock()
    mock_notebook_model = MagicMock()
    app.state.research_model = mock_research_model
    app.state.notebook_model = mock_notebook_model

    with (
        patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
        patch(
            "stellabook.fastapi_app.download_pdf",
            return_value=b"fake-pdf",
        ) as mock_download,
        patch(
            "stellabook.fastapi_app.extract_figures",
            return_value=[],
        ) as mock_extract,
        patch(
            "stellabook.fastapi_app.extract_text_from_pdf",
            return_value="Extracted paper text.",
        ) as mock_extract_text,
        patch(
            "stellabook.fastapi_app.research_paper",
            return_value="## Background\nSome research.",
        ) as mock_research,
        patch(
            "stellabook.fastapi_app.generate_notebook_content",
            return_value=mock_content,
        ) as mock_gen,
    ):
        mock_arxiv = AsyncMock()
        mock_arxiv.get_paper.return_value = _MOCK_PAPER
        mock_arxiv_cls.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv)
        mock_arxiv_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        yield PipelineMocks(
            arxiv_client=mock_arxiv_cls,
            download_pdf=mock_download,
            extract_figures=mock_extract,
            extract_text=mock_extract_text,
            research_paper=mock_research,
            generate_content=mock_gen,
            research_model=mock_research_model,
            notebook_model=mock_notebook_model,
        )


class TestHealthEndpoint:
    async def test_health_returns_ok(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestGenerateEndpoint:
    async def test_generate_returns_notebook(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/generate",
                json={"arxiv_id": "2301.07041"},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/x-ipynb+json"
        assert "2301.07041.ipynb" in response.headers["content-disposition"]

        nb_data = json.loads(response.text)
        assert nb_data["nbformat"] == 4
        # 3 cells: front matter + markdown + code
        assert len(nb_data["cells"]) == 3
        front_matter = "".join(nb_data["cells"][0]["source"])
        assert "Test Paper" in front_matter

        pipeline_mocks.download_pdf.assert_called_once_with(_MOCK_PAPER)
        pipeline_mocks.extract_figures.assert_called_once_with(
            _MOCK_PAPER, pdf_bytes=b"fake-pdf"
        )
        pipeline_mocks.extract_text.assert_called_once_with(b"fake-pdf")

        expected_ctx = PipelineContext(
            paper=_MOCK_PAPER,
            research_model=pipeline_mocks.research_model,
            notebook_model=pipeline_mocks.notebook_model,
            paper_text="Extracted paper text.",
            figures=[],
            interactive=False,
        )
        pipeline_mocks.research_paper.assert_called_once_with(expected_ctx)
        pipeline_mocks.generate_content.assert_called_once_with(
            expected_ctx, "## Background\nSome research."
        )

    async def test_generate_passes_interactive_flag(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/generate",
                json={"arxiv_id": "2301.07041", "interactive": True},
            )

        assert response.status_code == 200
        expected_ctx = PipelineContext(
            paper=_MOCK_PAPER,
            research_model=pipeline_mocks.research_model,
            notebook_model=pipeline_mocks.notebook_model,
            paper_text="Extracted paper text.",
            figures=[],
            interactive=True,
        )
        pipeline_mocks.generate_content.assert_called_once_with(
            expected_ctx, "## Background\nSome research."
        )

    async def test_generate_returns_404_for_unknown_paper(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        mock_arxiv = AsyncMock()
        mock_arxiv.get_paper.return_value = None
        pipeline_mocks.arxiv_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_arxiv
        )

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/generate",
                json={"arxiv_id": "0000.00000"},
            )

        assert response.status_code == 404
        assert response.json()["detail"] == "Paper not found"

    async def test_generate_degrades_when_text_extraction_fails(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        pipeline_mocks.extract_text.side_effect = RuntimeError("extraction failed")

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/generate",
                json={"arxiv_id": "2301.07041"},
            )

        assert response.status_code == 200
        expected_ctx = PipelineContext(
            paper=_MOCK_PAPER,
            research_model=pipeline_mocks.research_model,
            notebook_model=pipeline_mocks.notebook_model,
            paper_text=None,
            figures=[],
            interactive=False,
        )
        pipeline_mocks.research_paper.assert_called_once_with(expected_ctx)
        pipeline_mocks.generate_content.assert_called_once_with(
            expected_ctx, "## Background\nSome research."
        )

    async def test_generate_skips_text_extraction_when_no_pdf(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        pipeline_mocks.download_pdf.return_value = None

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/generate",
                json={"arxiv_id": "2301.07041"},
            )

        assert response.status_code == 200
        pipeline_mocks.extract_text.assert_not_called()
        expected_ctx = PipelineContext(
            paper=_MOCK_PAPER,
            research_model=pipeline_mocks.research_model,
            notebook_model=pipeline_mocks.notebook_model,
            paper_text=None,
            figures=[],
            interactive=False,
        )
        pipeline_mocks.research_paper.assert_called_once_with(expected_ctx)
        pipeline_mocks.generate_content.assert_called_once_with(
            expected_ctx, "## Background\nSome research."
        )
