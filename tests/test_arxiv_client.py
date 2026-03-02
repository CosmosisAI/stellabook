"""Tests for ArxivClient: XML parsing, search, and rate limiting."""

import time

import httpx

from stellabook.arxiv_client import ArxivClient, _parse_feed

from .conftest import SAMPLE_ATOM_FEED, SAMPLE_EMPTY_FEED


class TestAtomParsing:
    """Test XML parsing of Atom feeds."""

    def test_parse_paper_fields(self, sample_feed: str) -> None:
        result = _parse_feed(sample_feed)
        assert len(result.papers) == 1
        paper = result.papers[0]
        assert paper.arxiv_id == "2301.07041v1"
        assert paper.title == "Sample Paper Title"
        assert paper.summary == "This is a sample abstract for testing."
        assert paper.primary_category == "cs.AI"

    def test_parse_authors(self, sample_feed: str) -> None:
        paper = _parse_feed(sample_feed).papers[0]
        assert len(paper.authors) == 2
        assert paper.authors[0].name == "Alice Smith"
        assert paper.authors[0].affiliation == "MIT"
        assert paper.authors[1].name == "Bob Jones"
        assert paper.authors[1].affiliation is None

    def test_parse_categories(self, sample_feed: str) -> None:
        paper = _parse_feed(sample_feed).papers[0]
        terms = [c.term for c in paper.categories]
        assert terms == ["cs.AI", "cs.LG"]

    def test_parse_links_and_pdf_url(self, sample_feed: str) -> None:
        paper = _parse_feed(sample_feed).papers[0]
        assert len(paper.links) == 2
        assert paper.pdf_url == "http://arxiv.org/pdf/2301.07041v1"

    def test_parse_comment(self, sample_feed: str) -> None:
        paper = _parse_feed(sample_feed).papers[0]
        assert paper.comment == "10 pages, 3 figures"

    def test_parse_search_metadata(self, sample_feed: str) -> None:
        result = _parse_feed(sample_feed)
        assert result.total_results == 1
        assert result.start_index == 0
        assert result.items_per_page == 10

    def test_abstract_url_property(self, sample_feed: str) -> None:
        paper = _parse_feed(sample_feed).papers[0]
        assert paper.abstract_url == "https://arxiv.org/abs/2301.07041v1"

    def test_empty_feed_filters_error_entries(self, empty_feed: str) -> None:
        result = _parse_feed(empty_feed)
        assert result.total_results == 0
        assert len(result.papers) == 0

    def test_parse_timestamps(self, sample_feed: str) -> None:
        paper = _parse_feed(sample_feed).papers[0]
        assert paper.published.year == 2023
        assert paper.updated.month == 1


class TestSearch:
    """Test ArxivClient.search and get_paper using mocked HTTP."""

    async def test_search_returns_papers(self, client: ArxivClient) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=SAMPLE_ATOM_FEED)
        )
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=0.0,
        ) as c:
            result = await c.search("quantum computing")
            assert len(result.papers) == 1
            assert result.papers[0].arxiv_id == "2301.07041v1"

    async def test_get_paper_found(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=SAMPLE_ATOM_FEED)
        )
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=0.0,
        ) as c:
            paper = await c.get_paper("2301.07041v1")
            assert paper is not None
            assert paper.arxiv_id == "2301.07041v1"

    async def test_get_paper_not_found(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=SAMPLE_EMPTY_FEED)
        )
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=0.0,
        ) as c:
            paper = await c.get_paper("0000.00000")
            assert paper is None

    async def test_search_passes_query_params(self) -> None:
        captured_url: str = ""

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_url
            captured_url = str(request.url)
            return httpx.Response(200, text=SAMPLE_EMPTY_FEED)

        transport = httpx.MockTransport(handler)
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=0.0,
        ) as c:
            await c.search("all:electron", start=5, max_results=20)
            assert "search_query=all%3Aelectron" in captured_url
            assert "start=5" in captured_url
            assert "max_results=20" in captured_url

    async def test_max_results_clamped(self) -> None:
        captured_url: str = ""

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_url
            captured_url = str(request.url)
            return httpx.Response(200, text=SAMPLE_EMPTY_FEED)

        transport = httpx.MockTransport(handler)
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=0.0,
        ) as c:
            await c.search("test", max_results=999)
            assert "max_results=100" in captured_url


class TestRateLimiting:
    """Test that rate limiting enforces minimum delay between requests."""

    async def test_rate_limit_enforces_delay(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=SAMPLE_EMPTY_FEED)
        )
        delay = 0.1  # short delay for testing
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=delay,
        ) as c:
            start = time.monotonic()
            await c.search("first")
            await c.search("second")
            elapsed = time.monotonic() - start
            assert elapsed >= delay

    async def test_zero_rate_limit_is_fast(self) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=SAMPLE_EMPTY_FEED)
        )
        async with ArxivClient(
            http_client=httpx.AsyncClient(transport=transport),
            rate_limit_seconds=0.0,
        ) as c:
            start = time.monotonic()
            await c.search("a")
            await c.search("b")
            await c.search("c")
            elapsed = time.monotonic() - start
            assert elapsed < 0.5

    async def test_context_manager_closes_owned_client(self) -> None:
        async with ArxivClient(rate_limit_seconds=0.0) as c:
            assert c._http is not None
        assert c._http.is_closed
