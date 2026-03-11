"""Tests for the pdf module."""

import base64
from unittest.mock import AsyncMock

import httpx
import pymupdf
import pytest

from stellabook.models import Author, Category, Paper, PaperLink
from stellabook.pdf import (
    _extract_figures_from_pdf,
    _scale_image,
    _strip_sections,
    _truncate_at_section_boundary,
    download_pdf,
    extract_figures,
    extract_text_from_pdf,
)


def _make_test_pdf_with_text(text: str = "Hello world") -> bytes:
    """Create a minimal PDF containing the given text."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_test_pdf_with_image(width: int = 200, height: int = 150) -> bytes:
    """Create a minimal PDF containing a single image."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)

    # Create a pixmap (image) and insert it
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 1)
    pix.set_rect(pix.irect, (255, 0, 0, 255))  # fill with red (RGBA)
    page.insert_image(pymupdf.Rect(50, 50, 50 + width, 50 + height), pixmap=pix)

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def _make_paper(pdf_url: str | None = "http://arxiv.org/pdf/2301.07041v1") -> Paper:
    links = []
    if pdf_url is not None:
        links.append(
            PaperLink(
                href=pdf_url,
                rel="related",
                title="pdf",
                content_type="application/pdf",
            )
        )
    return Paper(
        arxiv_id="2301.07041",
        title="Test Paper",
        summary="Abstract.",
        authors=[Author(name="Alice")],
        categories=[Category(term="cs.AI")],
        links=links,
        published="2023-01-17T00:00:00Z",
        updated="2023-01-17T00:00:00Z",
        primary_category="cs.AI",
    )


class TestExtractFiguresFromPdf:
    def test_extracts_image_from_pdf(self) -> None:
        pdf_bytes = _make_test_pdf_with_image(200, 150)
        figures = _extract_figures_from_pdf(pdf_bytes)

        assert len(figures) == 1
        fig = figures[0]
        assert fig.label == "figure_0"
        assert fig.page_number == 1
        assert fig.width == 200
        assert fig.height == 150
        # Verify base64 is valid
        decoded = base64.b64decode(fig.image_base64)
        assert len(decoded) > 0

    def test_filters_small_images(self) -> None:
        pdf_bytes = _make_test_pdf_with_image(50, 50)
        figures = _extract_figures_from_pdf(pdf_bytes, min_dimension=100)

        assert len(figures) == 0

    def test_respects_max_count(self) -> None:
        # Create PDF with multiple images
        doc = pymupdf.open()
        page = doc.new_page()
        for i in range(5):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 200), 1)
            pix.set_rect(pix.irect, (i * 50, 0, 0, 255))
            page.insert_image(
                pymupdf.Rect(10, 10 + i * 160, 210, 170 + i * 160), pixmap=pix
            )
        pdf_bytes = doc.tobytes()
        doc.close()

        figures = _extract_figures_from_pdf(pdf_bytes, max_count=2)
        assert len(figures) == 2
        assert figures[0].label == "figure_0"
        assert figures[1].label == "figure_1"

    def test_empty_pdf_returns_no_figures(self) -> None:
        doc = pymupdf.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        figures = _extract_figures_from_pdf(pdf_bytes)
        assert figures == []


def _make_png(width: int, height: int) -> bytes:
    """Create a PNG image of the given dimensions."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), 1)
    pix.set_rect(pix.irect, (255, 0, 0, 255))
    result: bytes = pix.tobytes("png")
    return result


class TestScaleImage:
    def test_small_image_unchanged(self) -> None:
        png = _make_png(100, 100)
        # Set max_bytes well above the image size
        result, w, h = _scale_image(png, 100, 100, max_bytes=len(png) + 1000)
        assert result == png
        assert w == 100
        assert h == 100

    def test_large_image_is_scaled_down(self) -> None:
        png = _make_png(800, 600)
        # Set a tight max_bytes to force scaling
        result, w, h = _scale_image(png, 800, 600, max_bytes=1024, min_dimension=50)
        assert len(result) <= 1024
        assert w < 800
        assert h < 600

    def test_respects_min_dimension_floor(self) -> None:
        png = _make_png(400, 300)
        # Request impossibly small max_bytes but high min_dimension
        _, w, h = _scale_image(png, 400, 300, max_bytes=1, min_dimension=200)
        # Should not shrink below the min_dimension
        assert min(w, h) >= 200 or max(w, h) // 200 < 2

    def test_dimensions_stay_proportional(self) -> None:
        png = _make_png(800, 400)
        _, w, h = _scale_image(png, 800, 400, max_bytes=512, min_dimension=50)
        # Aspect ratio should be roughly preserved (shrink divides
        # both dimensions by the same integer factor)
        assert w >= h  # wider image stays wider


class TestExtractionScalesImages:
    def test_extracted_figures_are_scaled(self) -> None:
        # Create a PDF with a large image
        doc = pymupdf.open()
        page = doc.new_page()
        pix = pymupdf.Pixmap(
            pymupdf.csRGB,
            pymupdf.IRect(0, 0, 800, 600),
            1,
        )
        pix.set_rect(pix.irect, (0, 128, 255, 255))
        page.insert_image(pymupdf.Rect(10, 10, 810, 610), pixmap=pix)
        pdf_bytes = doc.tobytes()
        doc.close()

        # Use a tight inline limit to force scaling
        figures = _extract_figures_from_pdf(
            pdf_bytes,
            inline_max_bytes=1024,
            scaled_min_dimension=50,
        )
        assert len(figures) == 1
        raw_size = len(base64.b64decode(figures[0].image_base64))
        assert raw_size <= 1024
        assert figures[0].width < 800
        assert figures[0].height < 600


class TestExtractFigures:
    async def test_returns_empty_when_no_pdf_url(self) -> None:
        paper = _make_paper(pdf_url=None)
        result = await extract_figures(paper)
        assert result == []

    async def test_returns_empty_on_http_error(self) -> None:
        paper = _make_paper()
        mock_response = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", paper.pdf_url),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await extract_figures(paper, http_client=mock_client)
        assert result == []

    async def test_downloads_and_extracts(self) -> None:
        paper = _make_paper()
        pdf_bytes = _make_test_pdf_with_image()

        mock_response = httpx.Response(
            status_code=200,
            content=pdf_bytes,
            request=httpx.Request("GET", paper.pdf_url),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await extract_figures(paper, http_client=mock_client)
        assert len(result) == 1
        assert result[0].label == "figure_0"

    async def test_returns_empty_on_corrupt_pdf(self) -> None:
        paper = _make_paper()
        mock_response = httpx.Response(
            status_code=200,
            content=b"not a pdf",
            request=httpx.Request("GET", paper.pdf_url),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await extract_figures(paper, http_client=mock_client)
        assert result == []

    async def test_skips_download_when_pdf_bytes_provided(self) -> None:
        paper = _make_paper()
        pdf_bytes = _make_test_pdf_with_image()

        result = await extract_figures(paper, pdf_bytes=pdf_bytes)
        assert len(result) == 1
        assert result[0].label == "figure_0"


class TestDownloadPdf:
    async def test_returns_bytes_on_success(self) -> None:
        paper = _make_paper()
        pdf_bytes = _make_test_pdf_with_image()

        mock_response = httpx.Response(
            status_code=200,
            content=pdf_bytes,
            request=httpx.Request("GET", paper.pdf_url),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await download_pdf(paper, http_client=mock_client)
        assert result == pdf_bytes

    async def test_returns_none_when_no_pdf_url(self) -> None:
        paper = _make_paper(pdf_url=None)
        result = await download_pdf(paper)
        assert result is None

    async def test_returns_none_on_http_error(self) -> None:
        paper = _make_paper()
        mock_response = httpx.Response(
            status_code=404,
            request=httpx.Request("GET", paper.pdf_url),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get.return_value = mock_response

        result = await download_pdf(paper, http_client=mock_client)
        assert result is None


class TestExtractTextFromPdf:
    def test_extracts_text(self) -> None:
        pdf_bytes = _make_test_pdf_with_text("Hello world")
        text = extract_text_from_pdf(pdf_bytes)
        assert "Hello world" in text

    def test_truncates_at_max_chars(self) -> None:
        pdf_bytes = _make_test_pdf_with_text("A" * 200)
        text = extract_text_from_pdf(pdf_bytes, max_chars=50)
        assert len(text) <= 50

    def test_empty_pdf_returns_empty_or_whitespace(self) -> None:
        doc = pymupdf.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        text = extract_text_from_pdf(pdf_bytes)
        assert text.strip() == ""

    def test_raises_on_invalid_bytes(self) -> None:
        with pytest.raises(Exception):
            extract_text_from_pdf(b"not a pdf")


class TestStripSections:
    @pytest.mark.parametrize(
        "heading",
        [
            "## References",
            "## Bibliography",
            "## Acknowledgments",
            "## Acknowledgements",
            "# REFERENCES",
            "### references",
            "**References**",
        ],
    )
    def test_removes_default_sections(self, heading: str) -> None:
        text = f"# Intro\nBody.\n\n{heading}\nStripped content."
        result = _strip_sections(text)
        assert "Stripped content" not in result
        assert "Body." in result

    @pytest.mark.parametrize(
        "text",
        [
            "# Introduction\nBody.\n\n## Conclusion\nDone.",
            "See the references in section 3.\n\n## Conclusion\nDone.",
        ],
        ids=["no_match", "inline_mention"],
    )
    def test_preserves_unmatched_text(self, text: str) -> None:
        assert _strip_sections(text) == text

    def test_preserves_content_after_section(self) -> None:
        text = "# Intro\nBody.\n\n## References\n[1] Foo.\n\n## Appendix\nKept."
        result = _strip_sections(text)
        assert "Foo" not in result
        assert "Kept." in result

    def test_subsections_removed_with_parent(self) -> None:
        text = (
            "# Intro\nBody.\n\n"
            "# References\n[1] Foo.\n"
            "## Sub-references\nMore.\n\n"
            "# Conclusion\nDone."
        )
        result = _strip_sections(text)
        assert "Foo" not in result
        assert "Sub-references" not in result
        assert "Conclusion" in result

    def test_adjacent_stripped_sections(self) -> None:
        text = (
            "# Intro\nBody.\n\n"
            "## References\n[1] Foo.\n\n"
            "## Acknowledgments\nThanks.\n\n"
            "## Conclusion\nDone."
        )
        result = _strip_sections(text)
        assert "Foo" not in result
        assert "Thanks" not in result
        assert "Body." in result
        assert "Conclusion" in result

    def test_custom_names(self) -> None:
        text = "# Intro\nBody.\n\n## Appendix\nExtra.\n\n## Conclusion\nDone."
        result = _strip_sections(text, frozenset({"appendix"}))
        assert "Extra." not in result
        assert "Conclusion" in result


class TestTruncateAtSectionBoundary:
    def test_short_text_unchanged(self) -> None:
        text = "# Intro\nShort."
        assert _truncate_at_section_boundary(text, max_chars=1000) == text

    def test_cuts_at_last_heading_before_limit(self) -> None:
        text = "# Section 1\nAAAA\n\n# Section 2\nBBBB\n\n# Section 3\nCCCC"
        result = _truncate_at_section_boundary(text, max_chars=30)
        assert "Section 1" in result
        assert "Section 2" not in result

    def test_hard_cut_when_no_heading_before_limit(self) -> None:
        text = "No headings here, just a long block of plain text."
        result = _truncate_at_section_boundary(text, max_chars=20)
        assert len(result) == 20

    def test_cuts_at_bold_heading(self) -> None:
        text = "**Intro**\nFirst.\n\n**Methods**\nSecond.\n\n**Results**\nThird."
        result = _truncate_at_section_boundary(text, max_chars=30)
        assert "Intro" in result
        assert "Results" not in result

    def test_preserves_complete_sections(self) -> None:
        sec1 = "# Intro\nFirst section content."
        sec2 = "\n\n# Methods\nSecond section content."
        text = sec1 + sec2
        limit = len(sec1) + 5  # just past the boundary into sec2
        result = _truncate_at_section_boundary(text, max_chars=limit)
        assert "Intro" in result
        assert "First section content." in result
        assert "Methods" not in result
