"""Async arXiv API client with rate limiting."""

import asyncio
import time
from datetime import datetime
from types import TracebackType
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as DefusedET
import httpx

from stellabook.config import (
    ARXIV_API_BASE_URL,
    ARXIV_DEFAULT_MAX_RESULTS,
    ARXIV_MAX_RESULTS_LIMIT,
    ARXIV_RATE_LIMIT_SECONDS,
)
from stellabook.models import (
    Author,
    Category,
    Paper,
    PaperLink,
    SearchResult,
)

# Atom 1.0 + arXiv-specific namespaces
_ATOM_NS = "http://www.w3.org/2005/Atom"
_OPENSEARCH_NS = "http://a9.com/-/spec/opensearch/1.1/"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


def _tag(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def _text(element: Element, ns: str, tag: str) -> str:
    """Get text content of a child element, or empty string."""
    child = element.find(_tag(ns, tag))
    if child is not None and child.text:
        return child.text.strip()
    return ""


def _parse_entry(entry: Element) -> Paper:
    """Parse a single Atom <entry> into a Paper model."""
    # Extract arXiv ID from the <id> tag (URL like http://arxiv.org/abs/XXXX.XXXXX[vN])
    raw_id = _text(entry, _ATOM_NS, "id")
    arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id

    title = " ".join(_text(entry, _ATOM_NS, "title").split())
    summary = _text(entry, _ATOM_NS, "summary").strip()
    published = datetime.fromisoformat(_text(entry, _ATOM_NS, "published"))
    updated = datetime.fromisoformat(_text(entry, _ATOM_NS, "updated"))

    # Authors
    authors: list[Author] = []
    for author_el in entry.findall(_tag(_ATOM_NS, "author")):
        name = _text(author_el, _ATOM_NS, "name")
        affiliation_el = author_el.find(_tag(_ARXIV_NS, "affiliation"))
        affiliation = (
            affiliation_el.text.strip()
            if affiliation_el is not None and affiliation_el.text
            else None
        )
        if name:
            authors.append(Author(name=name, affiliation=affiliation))

    # Categories
    categories: list[Category] = []
    for cat_el in entry.findall(_tag(_ATOM_NS, "category")):
        term = cat_el.get("term")
        if term:
            categories.append(Category(term=term))

    # Primary category
    primary_cat_el = entry.find(_tag(_ARXIV_NS, "primary_category"))
    primary_category = (
        primary_cat_el.get("term", "") if primary_cat_el is not None else ""
    )

    # Links
    links: list[PaperLink] = []
    for link_el in entry.findall(_tag(_ATOM_NS, "link")):
        href = link_el.get("href", "")
        rel = link_el.get("rel", "")
        title_attr = link_el.get("title")
        content_type = link_el.get("type")
        if href:
            links.append(
                PaperLink(
                    href=href,
                    rel=rel,
                    title=title_attr,
                    content_type=content_type,
                )
            )

    # Optional fields
    comment = _text(entry, _ARXIV_NS, "comment") or None
    journal_ref = _text(entry, _ARXIV_NS, "journal_ref") or None
    doi = _text(entry, _ARXIV_NS, "doi") or None

    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        summary=summary,
        authors=authors,
        categories=categories,
        links=links,
        published=published,
        updated=updated,
        primary_category=primary_category,
        comment=comment,
        journal_ref=journal_ref,
        doi=doi,
    )


def _is_valid_entry(entry: Element) -> bool:
    """Check whether an Atom entry is a real paper (not an error/empty result)."""
    raw_id = _text(entry, _ATOM_NS, "id")
    # arXiv error entries don't have /abs/ in the id
    return "/abs/" in raw_id


def _parse_feed(xml_data: str) -> SearchResult:
    """Parse the full Atom feed into a SearchResult."""
    root = DefusedET.fromstring(xml_data)

    total_results = int(_text(root, _OPENSEARCH_NS, "totalResults") or "0")
    start_index = int(_text(root, _OPENSEARCH_NS, "startIndex") or "0")
    items_per_page = int(_text(root, _OPENSEARCH_NS, "itemsPerPage") or "0")

    papers: list[Paper] = []
    for entry in root.findall(_tag(_ATOM_NS, "entry")):
        if _is_valid_entry(entry):
            papers.append(_parse_entry(entry))

    return SearchResult(
        total_results=total_results,
        start_index=start_index,
        items_per_page=items_per_page,
        papers=papers,
    )


class ArxivClient:
    """Async arXiv API client with rate limiting."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        rate_limit_seconds: float = ARXIV_RATE_LIMIT_SECONDS,
    ) -> None:
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient()
        self._rate_limit_seconds = rate_limit_seconds
        self._lock = asyncio.Lock()
        self._last_request_time: float = 0.0

    async def __aenter__(self) -> "ArxivClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    async def _wait_for_rate_limit(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._rate_limit_seconds:
                await asyncio.sleep(self._rate_limit_seconds - elapsed)
            self._last_request_time = time.monotonic()

    async def _get(self, params: dict[str, str | int]) -> str:
        await self._wait_for_rate_limit()
        response = await self._http.get(ARXIV_API_BASE_URL, params=params)
        response.raise_for_status()
        return response.text

    async def search(
        self,
        query: str,
        *,
        start: int = 0,
        max_results: int = ARXIV_DEFAULT_MAX_RESULTS,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> SearchResult:
        """Search arXiv for papers matching a query."""
        max_results = min(max_results, ARXIV_MAX_RESULTS_LIMIT)
        params: dict[str, str | int] = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        xml_data = await self._get(params)
        return _parse_feed(xml_data)

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        """Fetch a single paper by its arXiv ID. Returns None if not found."""
        params: dict[str, str | int] = {
            "id_list": arxiv_id,
            "max_results": 1,
        }
        xml_data = await self._get(params)
        result = _parse_feed(xml_data)
        return result.papers[0] if result.papers else None
