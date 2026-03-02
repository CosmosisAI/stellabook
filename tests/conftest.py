"""Shared test fixtures."""

import pytest

from stellabook.arxiv_client import ArxivClient

SAMPLE_ATOM_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <opensearch:totalResults>1</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>10</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2301.07041v1</id>
    <updated>2023-01-17T18:58:29Z</updated>
    <published>2023-01-17T18:58:29Z</published>
    <title>Sample Paper Title</title>
    <summary>This is a sample abstract for testing.</summary>
    <author>
      <name>Alice Smith</name>
      <arxiv:affiliation>MIT</arxiv:affiliation>
    </author>
    <author>
      <name>Bob Jones</name>
    </author>
    <arxiv:comment>10 pages, 3 figures</arxiv:comment>
    <arxiv:primary_category term="cs.AI" />
    <category term="cs.AI" />
    <category term="cs.LG" />
    <link href="http://arxiv.org/abs/2301.07041v1" rel="alternate" type="text/html" />
    <link href="http://arxiv.org/pdf/2301.07041v1" rel="related"
          title="pdf" type="application/pdf" />
  </entry>
</feed>
"""

SAMPLE_EMPTY_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <opensearch:totalResults>0</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>10</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id</id>
    <title>Error</title>
    <summary>No results found</summary>
  </entry>
</feed>
"""


@pytest.fixture
def sample_feed() -> str:
    return SAMPLE_ATOM_FEED


@pytest.fixture
def empty_feed() -> str:
    return SAMPLE_EMPTY_FEED


@pytest.fixture
async def client() -> ArxivClient:
    """ArxivClient with rate limiting disabled for fast tests."""
    return ArxivClient(rate_limit_seconds=0.0)
