"""High-confidence table reconstruction for specification content."""

import re
from collections.abc import Sequence

import pymupdf

from .cleaner import clean_block_text
from .models import TextBlock


_STRONG_HEADER_RE = re.compile(
    r"(?:"
    r"item(?:\s+(?:number|no\.?))?"
    r"|specifications?"
    r"|(?:fill\s+)?capacity"
    r"|part\s+number"
    r"|dimension(?:\s+[a-z])?"
    r"|application"
    r"|quantity"
    r"|thickness(?:\s+\([^)]*\))?"
    r")",
    re.IGNORECASE,
)
_TORQUE_UNIT_RE = re.compile(r"^(?:n[·.]?m|nm|lb-ft|lb-in)$", re.IGNORECASE)
_TABLE_PAGE_HINT_RE = re.compile(
    r"\b(?:"
    r"item\s+(?:description|specification)"
    r"|part\s+number"
    r"|dimension\s+[a-z]"
    r"|description\s+(?:n[·.]?m|nm|lb-ft|lb-in)"
    r"|fill\s+capacity"
    r"|quantity"
    r")\b",
    re.IGNORECASE,
)


def _normalize_rows(
    rows: Sequence[Sequence[str | None]],
    column_count: int,
) -> tuple[tuple[str, ...], ...]:
    normalized_rows: list[tuple[str, ...]] = []

    for row in rows:
        cells = [clean_block_text(cell) if cell else "" for cell in row[:column_count]]
        cells.extend("" for _ in range(column_count - len(cells)))
        normalized_rows.append(tuple(cells))

    return tuple(normalized_rows)


def _has_high_confidence_header(rows: tuple[tuple[str, ...], ...]) -> bool:
    if len(rows) < 2 or len(rows[0]) < 2:
        return False

    header = rows[0]
    nonempty_header_cells = tuple(cell for cell in header if cell)
    if len(nonempty_header_cells) < 2:
        return False

    return any(_STRONG_HEADER_RE.fullmatch(cell) for cell in nonempty_header_cells) or any(
        _TORQUE_UNIT_RE.fullmatch(cell) for cell in nonempty_header_cells
    )


def _escape_markdown_cell(cell: str) -> str:
    return cell.replace("|", r"\|").replace("\n", "<br>")


def _to_markdown(rows: tuple[tuple[str, ...], ...]) -> str:
    header = tuple(cell or f"Column {index}" for index, cell in enumerate(rows[0], start=1))
    lines = [
        f"| {' | '.join(_escape_markdown_cell(cell) for cell in header)} |",
        f"| {' | '.join('---' for _ in header)} |",
    ]

    for row in rows[1:]:
        lines.append(f"| {' | '.join(_escape_markdown_cell(cell) for cell in row)} |")

    return "\n".join(lines)


def extract_specification_table_blocks(
    page: pymupdf.Page,
    *,
    page_text: str | None = None,
) -> tuple[TextBlock, ...]:
    """Convert reliable specification tables into reading-order text blocks.

    Tables without a recognizable specification header are deliberately left to
    raw block extraction. This fallback prevents uncertain layout reconstruction
    from damaging identifiers or associating values with the wrong columns.
    """
    if page_text is None:
        page_text = page.get_text("text")

    normalized_page_text = " ".join(page_text.split())
    if not _TABLE_PAGE_HINT_RE.search(normalized_page_text):
        return ()

    table_blocks: list[TextBlock] = []

    for table in page.find_tables().tables:
        rows = _normalize_rows(table.extract(), table.col_count)
        if not _has_high_confidence_header(rows):
            continue

        x0, y0, x1, y1 = table.bbox
        table_blocks.append(
            TextBlock(
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                text=_to_markdown(rows),
            )
        )

    return tuple(table_blocks)
