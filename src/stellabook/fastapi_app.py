"""Minimal FastAPI application for Stellabook."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from stellabook.arxiv_client import ArxivClient
from stellabook.config import get_notebook_model, get_research_model
from stellabook.figure_extractor import extract_figures
from stellabook.generator import generate_notebook_content, research_paper
from stellabook.notebook_builder import build_notebook, notebook_to_json
from stellabook.notebook_models import GenerateRequest

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.research_model = get_research_model()
    app.state.notebook_model = get_notebook_model()
    yield


app = FastAPI(title="Stellabook API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/generate")
async def generate(request: GenerateRequest) -> Response:
    async with ArxivClient() as arxiv:
        paper = await arxiv.get_paper(request.arxiv_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    figures = await extract_figures(paper)

    research = await research_paper(paper, model=app.state.research_model)
    content = await generate_notebook_content(
        paper, research, figures=figures, model=app.state.notebook_model,
        interactive=request.interactive,
    )
    nb = build_notebook(content, paper=paper, figures=figures)
    nb_json = notebook_to_json(nb)

    filename = f"{request.arxiv_id.replace('/', '_')}.ipynb"
    return Response(
        content=nb_json,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main() -> None:
    uvicorn.run("stellabook.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
