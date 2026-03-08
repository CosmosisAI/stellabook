"""Pure nbformat logic for building Jupyter notebooks."""

import ast
import re
import sys

import black
import nbformat
from black.parsing import InvalidInput as _BlackInvalidInput
from nbformat import NotebookNode

from stellabook.config import (
    FIGURE_INLINE_MAX_BYTES,
    NOTEBOOK_NBFORMAT_MINOR,
    NOTEBOOK_NBFORMAT_VERSION,
)
from stellabook.models import Paper
from stellabook.notebook_models import CellType, Figure, NotebookCell, NotebookContent

FIGURE_REF_RE = re.compile(r"!\[([^\]]*)\]\(attachment:(figure_\d+)\.png\)")

MODULE_TO_PACKAGE: dict[str, str] = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "gi": "PyGObject",
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
    "yaml": "PyYAML",
    "serial": "pyserial",
    "usb": "pyusb",
    "wx": "wxPython",
}


def extract_imports(cells: list[NotebookCell]) -> set[str]:
    """Extract third-party package names from code cells.

    Parses each code cell with ast to find import statements, filters out
    stdlib modules, and maps known module names to their pip package names.
    """
    stdlib = sys.stdlib_module_names
    top_level_modules: set[str] = set()

    for cell in cells:
        if cell.cell_type != CellType.CODE:
            continue
        try:
            tree = ast.parse(cell.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_level_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    top_level_modules.add(node.module.split(".")[0])

    third_party = top_level_modules - stdlib
    return {MODULE_TO_PACKAGE.get(m, m) for m in third_party}


def build_install_cell(packages: set[str]) -> NotebookNode | None:
    """Build a pip install code cell for the given packages.

    Returns None if the set is empty.
    """
    if not packages:
        return None
    pkgs = " ".join(sorted(packages))
    cell: NotebookNode = nbformat.v4.new_code_cell(f"!pip install {pkgs}")  # type: ignore[no-untyped-call]
    return cell


def format_code_cells(nb: NotebookNode) -> NotebookNode:
    """Format code cells in a notebook using Black.

    Skips non-code cells and shell commands (lines starting with ``!``).
    If Black cannot parse a cell, its source is left unchanged.
    """
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if cell.source.startswith("!"):
            continue
        # TODO(claude): Is this really what we want here? maybe we should
        # explore more options for this as more use cases build up.
        try:
            cell.source = black.format_str(cell.source, mode=black.Mode())
        except _BlackInvalidInput:
            continue
    return nb


def _embed_figures(
    cell: NotebookNode,
    figures_by_label: dict[str, Figure],
    inline_max_bytes: int = FIGURE_INLINE_MAX_BYTES,
) -> None:
    """Resolve figure references in a markdown cell.

    Small figures (base64 data <= *inline_max_bytes*) are inlined as
    ``<img src="data:image/png;base64,…">`` HTML tags so they render
    without a running Jupyter attachment server.

    Larger figures fall back to the nbformat attachment mechanism.
    """
    matches: list[tuple[str, str]] = FIGURE_REF_RE.findall(cell.source)
    if not matches:
        return

    attachments: dict[str, dict[str, str]] = {}

    for alt_text, label in matches:
        if label not in figures_by_label:
            continue
        fig = figures_by_label[label]
        raw_bytes = len(fig.image_base64) * 3 // 4
        if raw_bytes <= inline_max_bytes:
            # Replace the markdown image with an inline HTML <img>
            img_tag = (
                f'<center><img src="data:image/png;base64,'
                f'{fig.image_base64}" alt="{alt_text}" />'
                f"</center>"
            )
            cell.source = cell.source.replace(
                f"![{alt_text}](attachment:{label}.png)",
                img_tag,
            )
        else:
            attachments[f"{label}.png"] = {
                "image/png": fig.image_base64,
            }

    if attachments:
        cell["attachments"] = attachments


def build_front_matter_cell(paper: Paper) -> NotebookNode:
    """Build a deterministic markdown cell with paper metadata."""
    authors = ", ".join(a.name for a in paper.authors)
    categories = ", ".join(c.term for c in paper.categories)
    published = paper.published.strftime("%B %d, %Y")

    lines = [
        f"# {paper.title}",
        "",
        f"**Authors:** {authors}",
        "",
        f"**arXiv:** [{paper.arxiv_id}]({paper.abstract_url})",
        "",
        f"**Categories:** {categories}",
        "",
        f"**Published:** {published}",
    ]

    if paper.doi:
        lines += ["", f"**DOI:** {paper.doi}"]
    if paper.journal_ref:
        lines += ["", f"**Journal Ref:** {paper.journal_ref}"]

    cell: NotebookNode = nbformat.v4.new_markdown_cell("\n".join(lines))  # type: ignore[no-untyped-call]
    return cell


def build_notebook(
    content: NotebookContent,
    *,
    paper: Paper | None = None,
    figures: list[Figure] | None = None,
) -> NotebookNode:
    """Create a NotebookNode from structured notebook content."""
    nb: NotebookNode = nbformat.v4.new_notebook()  # type: ignore[no-untyped-call]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.nbformat = NOTEBOOK_NBFORMAT_VERSION
    nb.nbformat_minor = NOTEBOOK_NBFORMAT_MINOR

    figures_by_label: dict[str, Figure] = {}
    if figures:
        figures_by_label = {f.label: f for f in figures}

    packages = extract_imports(content.cells)
    install_cell = build_install_cell(packages)
    if install_cell is not None:
        nb.cells.append(install_cell)

    if paper is not None:
        nb.cells.append(build_front_matter_cell(paper))

    for cell in content.cells:
        match cell.cell_type:
            case CellType.MARKDOWN:
                nb_cell: NotebookNode = nbformat.v4.new_markdown_cell(cell.source)  # type: ignore[no-untyped-call]
                if figures_by_label:
                    _embed_figures(nb_cell, figures_by_label)
                nb.cells.append(nb_cell)
            case CellType.CODE:
                nb.cells.append(nbformat.v4.new_code_cell(cell.source))  # type: ignore[no-untyped-call]
            case _:
                raise NotImplementedError(f"Unsupported cell type: {cell.cell_type}")

    format_code_cells(nb)
    return nb


def notebook_to_json(nb: NotebookNode) -> str:
    """Serialize a NotebookNode to a JSON string."""
    result: str = nbformat.writes(nb)  # type: ignore[no-untyped-call]
    return result
