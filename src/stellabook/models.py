"""Pydantic models for arXiv paper data."""

from datetime import datetime

from pydantic import BaseModel


class Author(BaseModel):
    """An author of an arXiv paper."""

    name: str
    affiliation: str | None = None


class Category(BaseModel):
    """An arXiv category tag."""

    term: str


class PaperLink(BaseModel):
    """A link associated with an arXiv paper."""

    href: str
    rel: str
    title: str | None = None
    content_type: str | None = None


class Paper(BaseModel):
    """A parsed arXiv paper entry."""

    arxiv_id: str
    title: str
    summary: str
    authors: list[Author]
    categories: list[Category]
    links: list[PaperLink]
    published: datetime
    updated: datetime
    primary_category: str
    comment: str | None = None
    journal_ref: str | None = None
    doi: str | None = None

    @property
    def pdf_url(self) -> str | None:
        # TODO(claude): Need to double check if this makes sense.
        # Can we assume that the first pdf link is correct?
        for link in self.links:
            if link.title == "pdf":
                return link.href
        return None

    @property
    def abstract_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"


class SearchResult(BaseModel):
    """Result of an arXiv API search query."""

    total_results: int
    start_index: int
    items_per_page: int
    papers: list[Paper]
