"""Extract figures and text from arXiv paper PDFs using PyMuPDF."""

import base64
import logging
import math
import re
from typing import Any

import httpx
import logfire
import pymupdf
import pymupdf4llm  # type: ignore[import-untyped]

from stellabook.config import (
    FIGURE_INLINE_MAX_BYTES,
    FIGURE_MAX_COUNT,
    FIGURE_MIN_DIMENSION,
    FIGURE_SCALED_MIN_DIMENSION,
    PAPER_TEXT_MAX_CHARS,
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

    pix: Any = pymupdf.Pixmap(image_bytes)

    # Estimate the shrink factor needed.  PNG size doesn't scale
    # linearly with pixel count, but area is a reasonable proxy.
    ratio = len(image_bytes) / max_bytes
    factor = max(2, math.ceil(math.sqrt(ratio)))

    while factor <= max(width, height) // min_dimension:
        shrunk: Any = pymupdf.Pixmap(pix, 0)  # copy
        shrunk.shrink(factor)
        png_bytes: bytes = shrunk.tobytes("png")
        new_w: int = shrunk.width
        new_h: int = shrunk.height

        if len(png_bytes) <= max_bytes:
            return png_bytes, new_w, new_h

        factor += 1

    # Could not fit under the limit without going below min_dimension.
    # Return the smallest version we produced.
    shrunk = pymupdf.Pixmap(pix, 0)
    max_factor = max(1, max(width, height) // min_dimension)
    shrunk.shrink(max_factor)
    return (
        shrunk.tobytes("png"),
        int(shrunk.width),
        int(shrunk.height),
    )


def _to_png(image_bytes: bytes, ext: str) -> bytes:
    """Convert image bytes to PNG if not already in that format."""
    if ext == "png":
        return image_bytes
    pix: Any = pymupdf.Pixmap(image_bytes)
    result: bytes = pix.tobytes("png")
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
    doc: Any = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    try:
        for page_num in range(len(doc)):
            if len(figures) >= max_count:
                break
            page: Any = doc[page_num]
            image_list: list[tuple[Any, ...]] = page.get_images(full=True)

            for _, img_info in enumerate(image_list):
                if len(figures) >= max_count:
                    break

                xref: int = img_info[0]
                base_image: dict[str, Any] | None = doc.extract_image(xref)
                if base_image is None:
                    continue

                width: int = base_image["width"]
                height: int = base_image["height"]

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
        doc.close()

    return figures


async def download_pdf(
    paper: Paper,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> bytes | None:
    """Download a paper's PDF, returning raw bytes or ``None`` on failure."""
    pdf_url = paper.pdf_url
    if pdf_url is None:
        logger.info(
            "No PDF URL for paper %s, skipping PDF download",
            paper.arxiv_id,
        )
        return None

    owns_client = http_client is None
    if http_client is None:
        http_client = httpx.AsyncClient()

    try:
        with logfire.span("download PDF", arxiv_id=paper.arxiv_id):
            response = await http_client.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError:
        logger.warning(
            "Failed to download PDF for paper %s",
            paper.arxiv_id,
            exc_info=True,
        )
        return None
    finally:
        if owns_client:
            await http_client.aclose()


_ATX_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BOLD_HEADING_RE = re.compile(r"^\*\*(.+?)\*\*\s*$", re.MULTILINE)

STRIPPED_SECTION_NAMES: frozenset[str] = frozenset(
    {
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
    }
)


def _parse_headings(text: str) -> list[tuple[int, int, str]]:
    """Return ``(position, level, name)`` for every heading in *text*.

    ATX headings (``# …``) carry their natural level.  Bold-line
    headings (``**…**``) are treated as level 1.
    """
    headings: list[tuple[int, int, str]] = []
    for m in _ATX_HEADING_RE.finditer(text):
        headings.append((m.start(), len(m.group(1)), m.group(2).lower()))
    for m in _BOLD_HEADING_RE.finditer(text):
        headings.append((m.start(), 1, m.group(1).lower()))
    headings.sort()
    return headings


def _strip_sections(
    text: str,
    names: frozenset[str] = STRIPPED_SECTION_NAMES,
) -> str:
    """Remove markdown sections whose heading matches a name in *names*.

    A section spans from its heading to the next heading of equal or
    higher level (or the end of the document).  Content after the
    removed section is preserved.
    """
    headings = _parse_headings(text)

    # Collect char-ranges to remove (reversed so splicing is safe).
    to_remove: list[tuple[int, int]] = []
    for i, (start, level, name) in enumerate(headings):
        if name not in names:
            continue
        end = len(text)
        for next_start, next_level, _ in headings[i + 1 :]:
            if next_level <= level:
                end = next_start
                break
        to_remove.append((start, end))

    for start, end in reversed(to_remove):
        text = text[:start] + text[end:]

    return text.rstrip()


def _truncate_at_section_boundary(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, preferring a section break.

    If the text already fits, it is returned unchanged.  Otherwise the
    last heading boundary that falls within the limit is used as the
    cut-off so the LLM receives complete sections rather than a
    mid-sentence break.  Falls back to a hard cut when no heading
    appears before the limit.
    """
    if len(text) <= max_chars:
        return text

    headings = _parse_headings(text)
    # Find the last heading that starts within the budget.
    cut = 0
    for pos, _, _ in headings:
        if pos <= max_chars:
            cut = pos
        else:
            break

    if cut > 0:
        return text[:cut].rstrip()
    return text[:max_chars]


def extract_text_from_pdf(
    pdf_bytes: bytes,
    *,
    max_chars: int = PAPER_TEXT_MAX_CHARS,
) -> str:
    """Extract markdown-formatted text from PDF bytes.

    Uses ``pymupdf4llm`` for LLM-optimised markdown extraction.
    Low-value sections (e.g. references, bibliography) are stripped,
    then the result is truncated at the nearest section boundary that
    fits within *max_chars*.
    """
    doc: Any = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        text = str(pymupdf4llm.to_markdown(doc))  # type: ignore[reportUnknownMemberType]
    finally:
        doc.close()
    text = _strip_sections(text)
    return _truncate_at_section_boundary(text, max_chars)


async def extract_figures(
    paper: Paper,
    *,
    pdf_bytes: bytes | None = None,
    http_client: httpx.AsyncClient | None = None,
    min_dimension: int = FIGURE_MIN_DIMENSION,
    max_count: int = FIGURE_MAX_COUNT,
) -> list[Figure]:
    """Download a paper's PDF and extract figures from it.

    When *pdf_bytes* is provided the download step is skipped.
    Returns an empty list if the paper has no PDF URL or if
    downloading/extraction fails.
    """
    if pdf_bytes is None:
        pdf_bytes = await download_pdf(paper, http_client=http_client)
        if pdf_bytes is None:
            return []

    try:
        with logfire.span(
            "parse figures from PDF", arxiv_id=paper.arxiv_id
        ) as parse_span:
            figures = _extract_figures_from_pdf(
                pdf_bytes,
                min_dimension=min_dimension,
                max_count=max_count,
            )
            parse_span.set_attribute("figure_count", len(figures))
            return figures
    except Exception:
        logger.warning(
            "Failed to extract figures from PDF for paper %s",
            paper.arxiv_id,
            exc_info=True,
        )
        return []
