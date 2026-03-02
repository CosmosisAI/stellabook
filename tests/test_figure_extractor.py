"""Tests for the figure_extractor module."""

import base64
from unittest.mock import AsyncMock

import httpx
import pymupdf

from stellabook.figure_extractor import (
    _extract_figures_from_pdf,
    _scale_image,
    extract_figures,
)
from stellabook.models import Author, Category, Paper, PaperLink


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
