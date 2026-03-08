"""Tests for the FastAPI application."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import ASGITransport, AsyncClient

from stellabook.fastapi_app import app
from stellabook.models import Author, Category, Paper
from stellabook.notebook_models import CellType, NotebookCell, NotebookContent

_MOCK_PAPER = Paper(
    arxiv_id="2301.07041",
    title="Test Paper",
    summary="A test abstract.",
    authors=[Author(name="Alice"), Author(name="Bob")],
    categories=[Category(term="cs.AI")],
    links=[],
    published=datetime(2023, 1, 17, tzinfo=timezone.utc),
    updated=datetime(2023, 1, 17, tzinfo=timezone.utc),
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

        mock_research_model = MagicMock()
        mock_notebook_model = MagicMock()
        app.state.research_model = mock_research_model
        app.state.notebook_model = mock_notebook_model

        with (
            patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
            patch(
                "stellabook.fastapi_app.extract_figures",
                return_value=mock_figures,
            ) as mock_extract,
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

        mock_extract.assert_called_once_with(mock_paper)
        mock_research_fn.assert_called_once_with(
            mock_paper, model=mock_research_model
        )
        mock_gen.assert_called_once_with(
            mock_paper, mock_research, figures=mock_figures,
            model=mock_notebook_model, interactive=False,
        )

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

        mock_research_model = MagicMock()
        mock_notebook_model = MagicMock()
        app.state.research_model = mock_research_model
        app.state.notebook_model = mock_notebook_model

        with (
            patch("stellabook.fastapi_app.ArxivClient") as mock_arxiv_cls,
            patch(
                "stellabook.fastapi_app.extract_figures",
                return_value=mock_figures,
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
        mock_gen.assert_called_once_with(
            mock_paper, mock_research, figures=mock_figures,
            model=mock_notebook_model, interactive=True,
        )

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
