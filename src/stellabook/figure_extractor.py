"""Extract figures from arXiv paper PDFs using PyMuPDF."""

import base64
import logging
import math

import httpx
import pymupdf

from stellabook.config import (
    FIGURE_INLINE_MAX_BYTES,
    FIGURE_MAX_COUNT,
    FIGURE_MIN_DIMENSION,
    FIGURE_SCALED_MIN_DIMENSION,
)
from stellabook.models import Paper
from stellabook.notebook_models import Figure

logger = logging.getLogger(__name__)


def _scale_image(
    image_bytes: bytes,
    width: int,
    height: int,
    *,
    max_bytes: int = FIGURE_INLINE_MAX_BYTES,
    min_dimension: int = FIGURE_SCALED_MIN_DIMENSION,
) -> tuple[bytes, int, int]:
    """Scale a PNG image down so its encoded size fits under *max_bytes*.

    Uses progressively larger shrink factors until the image fits or
    the *min_dimension* floor is reached.  Returns the (possibly
    unchanged) PNG bytes together with the final width and height.
    """
    if len(image_bytes) <= max_bytes:
        return image_bytes, width, height

    pix = pymupdf.Pixmap(image_bytes)  # type: ignore[no-untyped-call]

    # Estimate the shrink factor needed.  PNG size doesn't scale
    # linearly with pixel count, but area is a reasonable proxy.
    ratio = len(image_bytes) / max_bytes
    factor = max(2, math.ceil(math.sqrt(ratio)))

    while factor <= max(width, height) // min_dimension:
        shrunk = pymupdf.Pixmap(pix, 0)  # type: ignore[no-untyped-call]  # copy
        shrunk.shrink(factor)  # type: ignore[no-untyped-call]
        png_bytes: bytes = shrunk.tobytes("png")  # type: ignore[no-untyped-call]
        new_w = shrunk.width
        new_h = shrunk.height

        if len(png_bytes) <= max_bytes:
            return png_bytes, new_w, new_h

        factor += 1

    # Could not fit under the limit without going below min_dimension.
    # Return the smallest version we produced.
    shrunk = pymupdf.Pixmap(pix, 0)  # type: ignore[no-untyped-call]
    max_factor = max(1, max(width, height) // min_dimension)
    shrunk.shrink(max_factor)  # type: ignore[no-untyped-call]
    return (
        shrunk.tobytes("png"),  # type: ignore[no-untyped-call]
        shrunk.width,
        shrunk.height,
    )


def _to_png(image_bytes: bytes, ext: str) -> bytes:
    """Convert image bytes to PNG if not already in that format."""
    if ext == "png":
        return image_bytes
    pix = pymupdf.Pixmap(image_bytes)  # type: ignore[no-untyped-call]
    result: bytes = pix.tobytes("png")  # type: ignore[no-untyped-call]
    return result


def _extract_figures_from_pdf(
    pdf_bytes: bytes,
    *,
    min_dimension: int = FIGURE_MIN_DIMENSION,
    max_count: int = FIGURE_MAX_COUNT,
    inline_max_bytes: int = FIGURE_INLINE_MAX_BYTES,
    scaled_min_dimension: int = FIGURE_SCALED_MIN_DIMENSION,
) -> list[Figure]:
    """Extract images from PDF bytes, filtering out small/decorative ones."""
    figures: list[Figure] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]

    try:
        for page_num in range(len(doc)):
            if len(figures) >= max_count:
                break
            page = doc[page_num]
            image_list = page.get_images(full=True)  # type: ignore[attr-defined]

            for img_index, img_info in enumerate(image_list):
                if len(figures) >= max_count:
                    break

                xref = img_info[0]
                base_image = doc.extract_image(xref)  # type: ignore[no-untyped-call]
                if base_image is None:
                    continue

                width = base_image["width"]
                height = base_image["height"]

                if width < min_dimension or height < min_dimension:
                    continue

                image_bytes = _to_png(base_image["image"], base_image["ext"])

                image_bytes, width, height = _scale_image(
                    image_bytes,
                    width,
                    height,
                    max_bytes=inline_max_bytes,
                    min_dimension=scaled_min_dimension,
                )

                label = f"figure_{len(figures)}"
                figures.append(
                    Figure(
                        label=label,
                        image_base64=base64.b64encode(image_bytes).decode(),
                        page_number=page_num + 1,
                        width=width,
                        height=height,
                    )
                )
    finally:
        doc.close()  # type: ignore[no-untyped-call]

    return figures


async def extract_figures(
    paper: Paper,
    *,
    http_client: httpx.AsyncClient | None = None,
    min_dimension: int = FIGURE_MIN_DIMENSION,
    max_count: int = FIGURE_MAX_COUNT,
) -> list[Figure]:
    """Download a paper's PDF and extract figures from it.

    Returns an empty list if the paper has no PDF URL or if
    downloading/extraction fails.
    """
    pdf_url = paper.pdf_url
    if pdf_url is None:
        logger.info(
            "No PDF URL for paper %s, skipping figure extraction",
            paper.arxiv_id,
        )
        return []

    owns_client = http_client is None
    if http_client is None:
        http_client = httpx.AsyncClient()

    try:
        response = await http_client.get(pdf_url, follow_redirects=True)
        response.raise_for_status()
        pdf_bytes = response.content
    except httpx.HTTPError:
        logger.warning(
            "Failed to download PDF for paper %s", paper.arxiv_id, exc_info=True
        )
        return []
    finally:
        if owns_client:
            await http_client.aclose()

    try:
        return _extract_figures_from_pdf(
            pdf_bytes, min_dimension=min_dimension, max_count=max_count
        )
    except Exception:
        logger.warning(
            "Failed to extract figures from PDF for paper %s",
            paper.arxiv_id,
            exc_info=True,
        )
        return []
