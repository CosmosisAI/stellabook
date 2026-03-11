"""Tests for the FastAPI application."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
    async def test_generate_returns_notebook(self) -> None:
        mock_content = NotebookContent(
            title="Test Notebook",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Test"),
                NotebookCell(cell_type=CellType.CODE, source="x = 1"),
            ],
        )
        mock_paper = _MOCK_PAPER
        mock_research = "## Background\nSome research."
        mock_figures: list[object] = []
        mock_pdf_bytes = b"fake-pdf"
        mock_paper_text = "Extracted paper text."

        mock_research_model = MagicMock()
        mock_notebook_model = MagicMock()
        app.state.research_model = mock_research_model
        app.state.notebook_model = mock_notebook_model

        with (
            patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
            patch(
                "stellabook.fastapi_app.download_pdf",
                return_value=mock_pdf_bytes,
            ) as mock_download,
            patch(
                "stellabook.fastapi_app.extract_figures",
                return_value=mock_figures,
            ) as mock_extract,
            patch(
                "stellabook.fastapi_app.extract_text_from_pdf",
                return_value=mock_paper_text,
            ) as mock_extract_text,
            patch(
                "stellabook.fastapi_app.research_paper",
                return_value=mock_research,
            ) as mock_research_fn,
            patch(
                "stellabook.fastapi_app.generate_notebook_content",
                return_value=mock_content,
            ) as mock_gen,
        ):
            mock_arxiv = AsyncMock()
            mock_arxiv.get_paper.return_value = mock_paper
            mock_arxiv_cls.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv)
            mock_arxiv_cls.return_value.__aexit__ = AsyncMock(return_value=False)

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

        mock_download.assert_called_once_with(mock_paper)
        mock_extract.assert_called_once_with(mock_paper, pdf_bytes=mock_pdf_bytes)
        mock_extract_text.assert_called_once_with(mock_pdf_bytes)

        expected_ctx = PipelineContext(
            paper=mock_paper,
            research_model=mock_research_model,
            notebook_model=mock_notebook_model,
            paper_text=mock_paper_text,
            figures=mock_figures,  # type: ignore[arg-type]
            interactive=False,
        )
        mock_research_fn.assert_called_once_with(expected_ctx)
        mock_gen.assert_called_once_with(expected_ctx, mock_research)

    async def test_generate_passes_interactive_flag(self) -> None:
        mock_content = NotebookContent(
            title="Test Notebook",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Test"),
            ],
        )
        mock_paper = _MOCK_PAPER
        mock_research = "## Background\nSome research."
        mock_figures: list[object] = []
        mock_pdf_bytes = b"fake-pdf"
        mock_paper_text = "Extracted paper text."

        mock_research_model = MagicMock()
        mock_notebook_model = MagicMock()
        app.state.research_model = mock_research_model
        app.state.notebook_model = mock_notebook_model

        with (
            patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
            patch(
                "stellabook.fastapi_app.download_pdf",
                return_value=mock_pdf_bytes,
            ),
            patch(
                "stellabook.fastapi_app.extract_figures",
                return_value=mock_figures,
            ),
            patch(
                "stellabook.fastapi_app.extract_text_from_pdf",
                return_value=mock_paper_text,
            ),
            patch(
                "stellabook.fastapi_app.research_paper",
                return_value=mock_research,
            ),
            patch(
                "stellabook.fastapi_app.generate_notebook_content",
                return_value=mock_content,
            ) as mock_gen,
        ):
            mock_arxiv = AsyncMock()
            mock_arxiv.get_paper.return_value = mock_paper
            mock_arxiv_cls.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv)
            mock_arxiv_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    "/generate",
                    json={"arxiv_id": "2301.07041", "interactive": True},
                )

        assert response.status_code == 200
        expected_ctx = PipelineContext(
            paper=mock_paper,
            research_model=mock_research_model,
            notebook_model=mock_notebook_model,
            paper_text=mock_paper_text,
            figures=mock_figures,  # type: ignore[arg-type]
            interactive=True,
        )
        mock_gen.assert_called_once_with(expected_ctx, mock_research)

    async def test_generate_returns_404_for_unknown_paper(self) -> None:
        with patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls:
            mock_arxiv = AsyncMock()
            mock_arxiv.get_paper.return_value = None
            mock_arxiv_cls.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv)
            mock_arxiv_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    "/generate",
                    json={"arxiv_id": "0000.00000"},
                )

        assert response.status_code == 404
        assert response.json()["detail"] == "Paper not found"

    async def test_generate_degrades_when_text_extraction_fails(self) -> None:
        mock_content = NotebookContent(
            title="Test Notebook",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Test"),
            ],
        )
        mock_paper = _MOCK_PAPER
        mock_research = "## Background\nSome research."
        mock_figures: list[object] = []
        mock_pdf_bytes = b"fake-pdf"

        mock_research_model = MagicMock()
        mock_notebook_model = MagicMock()
        app.state.research_model = mock_research_model
        app.state.notebook_model = mock_notebook_model

        with (
            patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
            patch(
                "stellabook.fastapi_app.download_pdf",
                return_value=mock_pdf_bytes,
            ),
            patch(
                "stellabook.fastapi_app.extract_figures",
                return_value=mock_figures,
            ),
            patch(
                "stellabook.fastapi_app.extract_text_from_pdf",
                side_effect=RuntimeError("extraction failed"),
            ),
            patch(
                "stellabook.fastapi_app.research_paper",
                return_value=mock_research,
            ) as mock_research_fn,
            patch(
                "stellabook.fastapi_app.generate_notebook_content",
                return_value=mock_content,
            ) as mock_gen,
        ):
            mock_arxiv = AsyncMock()
            mock_arxiv.get_paper.return_value = mock_paper
            mock_arxiv_cls.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv)
            mock_arxiv_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    "/generate",
                    json={"arxiv_id": "2301.07041"},
                )

        assert response.status_code == 200
        expected_ctx = PipelineContext(
            paper=mock_paper,
            research_model=mock_research_model,
            notebook_model=mock_notebook_model,
            paper_text=None,
            figures=mock_figures,  # type: ignore[arg-type]
            interactive=False,
        )
        mock_research_fn.assert_called_once_with(expected_ctx)
        mock_gen.assert_called_once_with(expected_ctx, mock_research)

    async def test_generate_skips_text_extraction_when_no_pdf(self) -> None:
        mock_content = NotebookContent(
            title="Test Notebook",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Test"),
            ],
        )
        mock_paper = _MOCK_PAPER
        mock_research = "## Background\nSome research."
        mock_figures: list[object] = []

        mock_research_model = MagicMock()
        mock_notebook_model = MagicMock()
        app.state.research_model = mock_research_model
        app.state.notebook_model = mock_notebook_model

        with (
            patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
            patch(
                "stellabook.fastapi_app.download_pdf",
                return_value=None,
            ),
            patch(
                "stellabook.fastapi_app.extract_figures",
                return_value=mock_figures,
            ),
            patch(
                "stellabook.fastapi_app.extract_text_from_pdf",
            ) as mock_extract_text,
            patch(
                "stellabook.fastapi_app.research_paper",
                return_value=mock_research,
            ) as mock_research_fn,
            patch(
                "stellabook.fastapi_app.generate_notebook_content",
                return_value=mock_content,
            ) as mock_gen,
        ):
            mock_arxiv = AsyncMock()
            mock_arxiv.get_paper.return_value = mock_paper
            mock_arxiv_cls.return_value.__aenter__ = AsyncMock(return_value=mock_arxiv)
            mock_arxiv_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post(
                    "/generate",
                    json={"arxiv_id": "2301.07041"},
                )

        assert response.status_code == 200
        mock_extract_text.assert_not_called()
        expected_ctx = PipelineContext(
            paper=mock_paper,
            research_model=mock_research_model,
            notebook_model=mock_notebook_model,
            paper_text=None,
            figures=mock_figures,  # type: ignore[arg-type]
            interactive=False,
        )
        mock_research_fn.assert_called_once_with(expected_ctx)
        mock_gen.assert_called_once_with(expected_ctx, mock_research)
