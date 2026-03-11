"""Minimal FastAPI application for Stellabook."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import logfire
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from stellabook.arxiv_client import ArxivClient
from stellabook.config import get_notebook_model, get_research_model
from stellabook.figure_extractor import (
    download_pdf,
    extract_figures,
    extract_text_from_pdf,
)
from stellabook.generator import generate_notebook_content, research_paper
from stellabook.notebook_builder import build_notebook, notebook_to_json
from stellabook.notebook_models import GenerateRequest
from stellabook.observability import configure_observability

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_observability(app)
    app.state.research_model = get_research_model()
    app.state.notebook_model = get_notebook_model()
    yield


app = FastAPI(title="Stellabook API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/generate")
async def generate(request: GenerateRequest) -> Response:
    with logfire.span("fetch paper metadata", arxiv_id=request.arxiv_id):
        async with ArxivClient() as arxiv:
            paper = await arxiv.get_paper(request.arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    pdf_bytes = await download_pdf(paper)

    with logfire.span("extract figures", arxiv_id=paper.arxiv_id):
        figures = await extract_figures(paper, pdf_bytes=pdf_bytes)

    paper_text: str | None = None
    if pdf_bytes is not None:
        try:
            with logfire.span("extract text from PDF", arxiv_id=paper.arxiv_id):
                paper_text = await asyncio.to_thread(extract_text_from_pdf, pdf_bytes)
        except Exception:
            logger.warning(
                "Failed to extract text from PDF for paper %s",
                paper.arxiv_id,
                exc_info=True,
            )

    with logfire.span("research paper", arxiv_id=paper.arxiv_id):
        research = await research_paper(
            paper, model=app.state.research_model, paper_text=paper_text
        )

    with logfire.span(
        "generate notebook content",
        arxiv_id=paper.arxiv_id,
        interactive=request.interactive,
    ):
        content = await generate_notebook_content(
            paper,
            research,
            figures=figures,
            model=app.state.notebook_model,
            interactive=request.interactive,
            paper_text=paper_text,
        )

    with logfire.span(
        "build notebook",
        arxiv_id=paper.arxiv_id,
        cell_count=len(content.cells),
    ):
        nb = build_notebook(content, paper=paper, figures=figures)
        nb_json = notebook_to_json(nb)

    filename = f"{request.arxiv_id.replace('/', '_')}.ipynb"
    return Response(
        content=nb_json,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main() -> None:
    uvicorn.run(
        "stellabook.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
