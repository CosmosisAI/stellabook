"""AI-powered notebook content generation."""

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from stellabook.config import get_notebook_model, get_research_model
from stellabook.models import Paper
from stellabook.notebook_models import Figure, NotebookContent

RESEARCH_SYSTEM_PROMPT = """\
You are a research scientist. Analyze the given arXiv paper in depth.

Write your analysis in markdown using the following sections:

## Background and Motivation
What problem does this paper address? What gap in existing research \
does it fill?

## Objectives
What does the paper set out to achieve or prove?

## Key Insights
The 3-5 most important findings or results (use a bulleted list).

## Novel Approach
What is new about the methodology compared to prior work?

## Key Takeaways
3-5 learnings a reader should walk away with (use a bulleted list).

## Practical Applications
How can readers apply this work?

## Suggested Code Demonstrations
2-4 concepts from the paper that could be illustrated with Python \
code (use a bulleted list, e.g. "Demonstrate the attention mechanism \
with a small matrix example").
"""

NOTEBOOK_SYSTEM_PROMPT = """\
You are an expert science educator creating a Jupyter notebook. \
You will receive:
1. An arXiv paper's metadata
2. A detailed research analysis of that paper

Use the research analysis to structure the notebook. The analysis \
identifies the key concepts, insights, and suggests code \
demonstrations.

Return the notebook content as structured output with a title and \
list of cells.

Content guidelines:
- Start with a markdown cell: paper title, authors, and a brief \
intro explaining the paper's motivation
- For each key insight, create a markdown cell explaining it \
clearly, followed by a code cell when the concept can be \
illustrated with Python
- Use the suggested code demonstrations from the research as a \
guide for what to implement in code cells
- Code cells should be self-contained and runnable
- Use clear, accessible language suitable for graduate students
- Include LaTeX math notation (MathJax) where equations help \
explain concepts
- End with a summary cell covering key takeaways and practical \
applications

Figure embedding guidelines:
- If figures from the paper are listed below the research analysis, \
you may reference them in markdown cells
- Use the syntax ![description](attachment:figure_N.png) to embed \
a figure, where N is the figure number from the available figures list
- Only reference figures that are listed as available
- Each figure should only appear once in the notebook
- Only embed a figure when it is directly relevant to the \
surrounding text; do not force figures into unrelated sections
- Write your own caption that specifically describes what the \
single embedded figure shows and how it relates to the current \
discussion — do NOT reuse or adapt the paper's original captions, \
which may describe multiple sub-figures or panels that are not all \
present in the extracted image
- If you are unsure what a figure depicts, it is better to omit it \
than to guess incorrectly
"""


def _build_user_message(paper: Paper) -> str:
    """Format paper metadata into a user message for the AI model."""
    authors = ", ".join(a.name for a in paper.authors)
    categories = ", ".join(c.term for c in paper.categories)
    return (
        f"Title: {paper.title}\n"
        f"Authors: {authors}\n"
        f"Categories: {categories}\n"
        f"arXiv ID: {paper.arxiv_id}\n"
        f"URL: {paper.abstract_url}\n\n"
        f"Abstract:\n{paper.summary}"
    )


def _build_notebook_user_message(
    paper: Paper,
    research: str,
    figures: list[Figure] | None = None,
) -> str:
    """Format paper metadata and research analysis into a message."""
    paper_section = _build_user_message(paper)
    msg = f"{paper_section}\n\n---\n\nResearch Analysis:\n{research}"

    if figures:
        msg += "\n\n---\n\nAvailable Figures:\n"
        for fig in figures:
            msg += (
                f"- {fig.label}: page {fig.page_number}, "
                f"{fig.width}x{fig.height} pixels\n"
            )

    return msg


async def research_paper(
    paper: Paper,
    *,
    model: BaseChatModel | None = None,
) -> str:
    """Analyze a paper in depth, returning markdown research."""
    if model is None:
        model = get_research_model()
    response = await model.ainvoke([
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_message(paper)),
    ])
    return cast(str, response.content)  # type: ignore[reportUnknownMemberType]


async def generate_notebook_content(
    paper: Paper,
    research: str,
    *,
    figures: list[Figure] | None = None,
    model: BaseChatModel | None = None,
) -> NotebookContent:
    """Generate notebook content for a paper using structured output."""
    if model is None:
        model = get_notebook_model()
    structured_model = model.with_structured_output(
        NotebookContent, include_raw=True
    )
    result = await structured_model.ainvoke([
        SystemMessage(content=NOTEBOOK_SYSTEM_PROMPT),
        HumanMessage(content=_build_notebook_user_message(paper, research, figures)),
    ])
    assert isinstance(result, dict)

    if result["parsing_error"] is not None:
        raise ValueError(f"Failed to parse notebook content: {result['parsing_error']}")

    metadata: dict[str, object] = result["raw"].response_metadata  # type: ignore[union-attr]
    stop_reason = metadata.get("stop_reason") or metadata.get("finish_reason")
    if stop_reason == "max_tokens":
        raise ValueError(
            "Notebook generation was truncated due to max_tokens limit"
        )

    parsed = result["parsed"]
    assert isinstance(parsed, NotebookContent)
    return parsed
