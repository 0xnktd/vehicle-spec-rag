"""Deterministic cleanup for text extracted from the service manual."""

import re
from collections.abc import Iterable

from .models import PageRecord, TextBlock

PDF_CLEANER_VERSION = "1.0.0"

_MANUAL_TITLE = "2014 F-150 Workshop Manual"
_PAGE_LABEL_RE = re.compile(r"Page\s+\d+\s+sur\s+\d+", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PROCEDURE_REVISION_RE = re.compile(
    r"Procedure\s+revision\s+date:\s*\d{1,2}/\d{1,2}/\d{4}",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"SECTION\s+\d", re.IGNORECASE)
_HYPHENATED_WRAP_RE = re.compile(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.:;?!])")
_PDF_BULLET_RE = re.compile(r"^(?:z|\x84)\s+")
_NUMBERED_STEP_RE = re.compile(r"^\d+[.)](?:\s|$)")
_STRUCTURAL_LABEL_RE = re.compile(
    r"^(?:(?:Yes|No)\b|(?:NOTE|WARNING|CAUTION):)",
    re.IGNORECASE,
)


def _classification_lines(text: str) -> tuple[str, ...]:
    """Return whitespace-normalized lines for boilerplate classification."""
    return tuple(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _is_recurring_header(text: str) -> bool:
    lines = _classification_lines(text)
    return (
        bool(lines)
        and any(_PAGE_LABEL_RE.fullmatch(line) for line in lines)
        and all(
            _PAGE_LABEL_RE.fullmatch(line) or line == _MANUAL_TITLE for line in lines
        )
    )


def _is_recurring_footer(text: str) -> bool:
    lines = _classification_lines(text)
    return (
        bool(lines)
        and any(line.casefold().startswith("file:///") for line in lines)
        and all(
            _ISO_DATE_RE.fullmatch(line)
            or line.casefold().startswith("file:///")
            or line.casefold() == "repair4less"
            for line in lines
        )
    )


def _normalize_line(line: str) -> str:
    normalized = " ".join(line.strip().split())
    normalized = _PDF_BULLET_RE.sub("- ", normalized)
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)


def _starts_structural_item(line: str) -> bool:
    return (
        line.startswith("- ")
        or bool(_NUMBERED_STEP_RE.match(line))
        or bool(_STRUCTURAL_LABEL_RE.match(line))
    )


def _reflow_lines(lines: Iterable[str]) -> str:
    """Join visual wraps while retaining list, step, and paragraph boundaries."""
    groups: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            groups.append(" ".join(current))
            current.clear()

    for line in lines:
        if not line:
            flush()
            continue

        if _starts_structural_item(line):
            flush()
        current.append(line)

    flush()
    return "\n".join(groups)


def clean_block_text(text: str) -> str:
    """Clean one text block without discarding meaningful Unicode characters."""
    if _is_recurring_header(text) or _is_recurring_footer(text):
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHENATED_WRAP_RE.sub("-", text)

    raw_lines = tuple(line for line in text.split("\n"))
    section_block = any(_SECTION_RE.match(line.strip()) for line in raw_lines)
    normalized_lines: list[str] = []

    for raw_line in raw_lines:
        line = _normalize_line(raw_line)

        if section_block and (
            line == _MANUAL_TITLE or _PROCEDURE_REVISION_RE.fullmatch(line)
        ):
            continue

        normalized_lines.append(line)

    if section_block:
        return "\n".join(line for line in normalized_lines if line).strip()

    return _reflow_lines(normalized_lines).strip()


def _block_center_is_inside(block: TextBlock, container: TextBlock) -> bool:
    x0, y0, x1, y1 = block.bbox
    container_x0, container_y0, container_x1, container_y1 = container.bbox
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    return (
        container_x0 <= center_x <= container_x1
        and container_y0 <= center_y <= container_y1
    )


def _looks_like_heading(text: str) -> bool:
    return (
        "\n" not in text
        and 2 < len(text) <= 120
        and not _starts_structural_item(text)
        and not text.endswith((".", ",", ":", ";", "?", "!"))
    )


def _is_adjacent_duplicate_heading(previous: TextBlock, current: TextBlock) -> bool:
    previous_x0, _, _, previous_y1 = previous.bbox
    current_x0, current_y0, _, _ = current.bbox
    return (
        current.text == previous.text
        and _looks_like_heading(current.text)
        and abs(current_x0 - previous_x0) <= 20
        and current_y0 - previous_y1 <= 40
    )


def _deduplicate_adjacent_headings(
    blocks: Iterable[TextBlock],
) -> tuple[TextBlock, ...]:
    deduplicated: list[TextBlock] = []

    for block in blocks:
        if deduplicated and _is_adjacent_duplicate_heading(deduplicated[-1], block):
            continue
        deduplicated.append(block)

    return tuple(deduplicated)


def clean_page_record(
    page: PageRecord,
    table_blocks: Iterable[TextBlock] = (),
) -> PageRecord:
    """Return a cleaned copy of a page, replacing supported table regions."""
    tables = tuple(table_blocks)
    cleaned_blocks: list[TextBlock] = []

    for block in page.blocks:
        if any(_block_center_is_inside(block, table) for table in tables):
            continue

        cleaned_text = clean_block_text(block.text)
        if not cleaned_text:
            continue

        cleaned_blocks.append(TextBlock(bbox=block.bbox, text=cleaned_text))

    cleaned_blocks.extend(tables)
    cleaned_blocks.sort(key=lambda block: (block.bbox[1], block.bbox[0]))

    return PageRecord(
        pdf_page=page.pdf_page,
        width=page.width,
        height=page.height,
        blocks=_deduplicate_adjacent_headings(cleaned_blocks),
    )
