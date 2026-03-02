"""Pydantic models for notebook generation."""

from enum import StrEnum

from pydantic import BaseModel


class CellType(StrEnum):
    """Type of notebook cell."""

    MARKDOWN = "markdown"
    CODE = "code"


class NotebookCell(BaseModel):
    """A single cell in a generated notebook."""

    cell_type: CellType
    source: str


class NotebookContent(BaseModel):
    """Structured content for a generated notebook."""

    title: str
    cells: list[NotebookCell]


class Figure(BaseModel):
    """An extracted figure from a paper's PDF."""

    label: str
    image_base64: str
    page_number: int
    width: int
    height: int


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    arxiv_id: str
