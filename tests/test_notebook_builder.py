"""Tests for notebook_builder module."""

import json

import nbformat

from stellabook.notebook_builder import (
    build_install_cell,
    build_notebook,
    extract_imports,
    format_code_cells,
    notebook_to_json,
)
from stellabook.notebook_models import CellType, Figure, NotebookCell, NotebookContent


class TestBuildNotebook:
    def test_creates_valid_notebook_structure(self) -> None:
        content = NotebookContent(
            title="Test Notebook",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Hello"),
                NotebookCell(cell_type=CellType.CODE, source="print('hi')"),
            ],
        )
        nb = build_notebook(content)

        assert nb.nbformat == 4
        assert nb.nbformat_minor == 5
        assert len(nb.cells) == 2
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[0].source == "# Hello"
        assert nb.cells[1].cell_type == "code"
        assert nb.cells[1].source == 'print("hi")\n'

    def test_kernelspec_is_python3(self) -> None:
        content = NotebookContent(title="Test", cells=[])
        nb = build_notebook(content)

        assert nb.metadata["kernelspec"]["name"] == "python3"
        assert nb.metadata["kernelspec"]["language"] == "python"

    def test_empty_cells(self) -> None:
        content = NotebookContent(title="Empty", cells=[])
        nb = build_notebook(content)

        assert len(nb.cells) == 0
        nbformat.validate(nb)


class TestExtractImports:
    def test_filters_out_stdlib(self) -> None:
        cells = [
            NotebookCell(cell_type=CellType.CODE, source="import os\nimport json"),
        ]
        assert extract_imports(cells) == set()

    def test_detects_third_party(self) -> None:
        cells = [
            NotebookCell(cell_type=CellType.CODE, source="import numpy\nimport pandas"),
        ]
        assert extract_imports(cells) == {"numpy", "pandas"}

    def test_from_import_resolves_to_top_level(self) -> None:
        cells = [
            NotebookCell(
                cell_type=CellType.CODE, source="from numpy.linalg import norm"
            ),
        ]
        assert extract_imports(cells) == {"numpy"}

    def test_aliased_import(self) -> None:
        cells = [
            NotebookCell(cell_type=CellType.CODE, source="import numpy as np"),
        ]
        assert extract_imports(cells) == {"numpy"}

    def test_module_to_package_mapping(self) -> None:
        cells = [
            NotebookCell(
                cell_type=CellType.CODE,
                source=(
                    "import cv2\n"
                    "from sklearn.ensemble import RandomForestClassifier\n"
                    "from PIL import Image"
                ),
            ),
        ]
        result = extract_imports(cells)
        assert result == {"opencv-python", "scikit-learn", "Pillow"}

    def test_syntax_error_skipped(self) -> None:
        cells = [
            NotebookCell(cell_type=CellType.CODE, source="def (\nbad syntax"),
            NotebookCell(cell_type=CellType.CODE, source="import numpy"),
        ]
        assert extract_imports(cells) == {"numpy"}

    def test_ignores_markdown_cells(self) -> None:
        cells = [
            NotebookCell(cell_type=CellType.MARKDOWN, source="import numpy"),
        ]
        assert extract_imports(cells) == set()

    def test_detects_ipywidgets_import(self) -> None:
        cells = [
            NotebookCell(
                cell_type=CellType.CODE,
                source="from ipywidgets import interact",
            ),
        ]
        assert "ipywidgets" in extract_imports(cells)


class TestBuildInstallCell:
    def test_produces_pip_install_command(self) -> None:
        cell = build_install_cell({"numpy", "pandas"})
        assert cell is not None
        assert cell.source == "!pip install numpy pandas"

    def test_sorted_for_determinism(self) -> None:
        cell = build_install_cell({"z_pkg", "a_pkg", "m_pkg"})
        assert cell is not None
        assert cell.source == "!pip install a_pkg m_pkg z_pkg"

    def test_returns_none_for_empty_set(self) -> None:
        assert build_install_cell(set()) is None


class TestNotebookToJson:
    def test_json_roundtrip(self) -> None:
        content = NotebookContent(
            title="Roundtrip",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Test"),
                NotebookCell(cell_type=CellType.CODE, source="x = 1"),
            ],
        )
        nb = build_notebook(content)
        nb_json = notebook_to_json(nb)

        parsed = json.loads(nb_json)
        assert parsed["nbformat"] == 4
        assert len(parsed["cells"]) == 2

        restored = nbformat.reads(nb_json, as_version=4)
        assert restored.cells[0].source == "# Test"
        assert restored.cells[1].source == "x = 1\n"


class TestBuildNotebookInstallIntegration:
    def test_prepends_install_cell_for_third_party_imports(self) -> None:
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(cell_type=CellType.MARKDOWN, source="# Analysis"),
                NotebookCell(cell_type=CellType.CODE, source="import numpy as np"),
            ],
        )
        nb = build_notebook(content)

        assert len(nb.cells) == 3
        assert nb.cells[0].cell_type == "code"
        assert nb.cells[0].source == "!pip install numpy"
        assert nb.cells[1].cell_type == "markdown"
        assert nb.cells[2].cell_type == "code"

    def test_no_install_cell_for_stdlib_only(self) -> None:
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(cell_type=CellType.CODE, source="import os\nimport json"),
            ],
        )
        nb = build_notebook(content)

        assert len(nb.cells) == 1
        assert nb.cells[0].source == "import os\nimport json\n"


class TestFormatCodeCells:
    def test_reformats_messy_code(self) -> None:
        nb: nbformat.NotebookNode = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell("x=1"))
        format_code_cells(nb)
        assert nb.cells[0].source == "x = 1\n"

    def test_leaves_markdown_untouched(self) -> None:
        nb: nbformat.NotebookNode = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_markdown_cell("x=1"))
        format_code_cells(nb)
        assert nb.cells[0].source == "x=1"

    def test_skips_shell_commands(self) -> None:
        nb: nbformat.NotebookNode = nbformat.v4.new_notebook()
        nb.cells.append(nbformat.v4.new_code_cell("!pip install numpy pandas"))
        format_code_cells(nb)
        assert nb.cells[0].source == "!pip install numpy pandas"

    def test_handles_syntax_errors_gracefully(self) -> None:
        nb: nbformat.NotebookNode = nbformat.v4.new_notebook()
        bad_source = "def (\nbad syntax"
        nb.cells.append(nbformat.v4.new_code_cell(bad_source))
        format_code_cells(nb)
        assert nb.cells[0].source == bad_source

    def test_build_notebook_formats_code_cells(self) -> None:
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(cell_type=CellType.CODE, source="x=1"),
            ],
        )
        nb = build_notebook(content)
        assert nb.cells[0].source == "x = 1\n"


class TestFigureEmbedding:
    def _make_small_figure(self, label: str = "figure_0") -> Figure:
        """Create a figure whose base64 data is well under 10KB."""
        return Figure(
            label=label,
            image_base64="iVBORw0KGgoAAAANSUhEUg==",
            page_number=1,
            width=200,
            height=150,
        )

    def _make_large_figure(self, label: str = "figure_0") -> Figure:
        """Create a figure whose decoded size exceeds 80KB."""
        import base64

        data = base64.b64encode(b"\x00" * 100_000).decode()
        return Figure(
            label=label,
            image_base64=data,
            page_number=1,
            width=800,
            height=600,
        )

    def test_small_figure_is_inlined_as_img_tag(self) -> None:
        fig = self._make_small_figure()
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(
                    cell_type=CellType.MARKDOWN,
                    source="![A diagram](attachment:figure_0.png)",
                ),
            ],
        )
        nb = build_notebook(content, figures=[fig])

        cell = nb.cells[0]
        assert "data:image/png;base64," in cell.source
        assert fig.image_base64 in cell.source
        assert 'alt="A diagram"' in cell.source
        assert "attachment:" not in cell.source
        assert "attachments" not in cell

    def test_large_figure_uses_attachment(self) -> None:
        fig = self._make_large_figure()
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(
                    cell_type=CellType.MARKDOWN,
                    source="![Big chart](attachment:figure_0.png)",
                ),
            ],
        )
        nb = build_notebook(content, figures=[fig])

        cell = nb.cells[0]
        assert "attachments" in cell
        assert "figure_0.png" in cell["attachments"]
        att = cell["attachments"]["figure_0.png"]
        assert att["image/png"] == fig.image_base64
        # Source should be unchanged (still markdown attachment ref)
        assert "attachment:figure_0.png" in cell.source

    def test_no_changes_without_figures(self) -> None:
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(
                    cell_type=CellType.MARKDOWN,
                    source="![A diagram](attachment:figure_0.png)",
                ),
            ],
        )
        nb = build_notebook(content)

        cell = nb.cells[0]
        assert "attachments" not in cell or cell.get("attachments") is None
        assert cell.source == "![A diagram](attachment:figure_0.png)"

    def test_unmatched_reference_left_alone(self) -> None:
        fig = self._make_small_figure("figure_5")
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(
                    cell_type=CellType.MARKDOWN,
                    source="![A diagram](attachment:figure_0.png)",
                ),
            ],
        )
        nb = build_notebook(content, figures=[fig])

        cell = nb.cells[0]
        assert cell.source == "![A diagram](attachment:figure_0.png)"

    def test_multiple_small_figures_inlined(self) -> None:
        fig0 = self._make_small_figure("figure_0")
        fig1 = Figure(
            label="figure_1",
            image_base64="AAAA",
            page_number=2,
            width=300,
            height=200,
        )
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(
                    cell_type=CellType.MARKDOWN,
                    source=(
                        "![Fig 1](attachment:figure_0.png)\n"
                        "![Fig 2](attachment:figure_1.png)"
                    ),
                ),
            ],
        )
        nb = build_notebook(content, figures=[fig0, fig1])

        cell = nb.cells[0]
        assert "attachment:" not in cell.source
        assert fig0.image_base64 in cell.source
        assert fig1.image_base64 in cell.source

    def test_mixed_small_and_large_figures(self) -> None:
        small = self._make_small_figure("figure_0")
        large = self._make_large_figure("figure_1")
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(
                    cell_type=CellType.MARKDOWN,
                    source=(
                        "![Small](attachment:figure_0.png)\n"
                        "![Large](attachment:figure_1.png)"
                    ),
                ),
            ],
        )
        nb = build_notebook(content, figures=[small, large])

        cell = nb.cells[0]
        # Small figure inlined
        assert small.image_base64 in cell.source
        assert "attachment:figure_0" not in cell.source
        # Large figure kept as attachment
        assert "attachment:figure_1.png" in cell.source
        assert "attachments" in cell
        assert "figure_1.png" in cell["attachments"]

    def test_code_cells_unaffected_by_figures(self) -> None:
        fig = self._make_small_figure()
        content = NotebookContent(
            title="Test",
            cells=[
                NotebookCell(cell_type=CellType.CODE, source="x = 1"),
            ],
        )
        nb = build_notebook(content, figures=[fig])

        cell = nb.cells[0]
        assert "attachments" not in cell
