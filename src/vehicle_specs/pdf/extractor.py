"""Extract cleaned PDF pages and propagate their service-manual context."""

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import pymupdf

from .cleaner import clean_page_record
from .models import ContextualPageRecord, PageRecord, SectionContext, TextBlock
from .tables import extract_specification_table_blocks

PDF_PARSER_VERSION = "1.0.0"

_SECTION_HEADING_RE = re.compile(
    r"SECTION\s+(?P<section_id>\d{3}-\d{2}[A-Z]?):\s*(?P<section_title>.+)"
)
_CATEGORY_RE = re.compile(r"[A-Z][A-Z0-9 &/()—-]*")
_NUMBERED_ITEM_RE = re.compile(r"\d+[.)](?:\s|$)")
_GENERIC_SUBHEADINGS = frozenset(
    {
        "General Specifications",
        "Installation",
        "Material",
        "Removal",
        "Removal and Installation",
        "Special Tool(s)",
        "Torque Specifications",
    }
)
_NON_TITLE_PREFIXES = (
    "- ",
    "| ",
    "CAUTION:",
    "NOTE:",
    "NOTICE:",
    "WARNING:",
)


def _validate_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"PDF path is not a file: {path}")

    return path


def _validate_document(document: pymupdf.Document, path: Path) -> None:
    if not document.is_pdf:
        raise ValueError(f"File is not a PDF: {path}")
    if document.needs_pass:
        raise PermissionError(f"PDF requires a password: {path}")
    if document.page_count == 0:
        raise ValueError(f"PDF contains no pages: {path}")


def extract_page(page: pymupdf.Page) -> PageRecord:
    """Extract ordered text blocks and their coordinates from one PDF page."""
    text_blocks: list[TextBlock] = []

    for block in page.get_text(  # type: ignore[no-untyped-call]
        "blocks", sort=True
    ):
        x0, y0, x1, y1, text, _, block_type = block

        if block_type != 0 or not text.strip():
            continue

        text_blocks.append(
            TextBlock(
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                text=text,
            )
        )

    page_rect = page.rect
    return PageRecord(
        pdf_page=page.number + 1,
        width=float(page_rect.width),
        height=float(page_rect.height),
        blocks=tuple(text_blocks),
    )


def extract_clean_page(page: pymupdf.Page) -> PageRecord:
    """Extract and clean a page, reconstructing reliable specification tables."""
    raw_page = extract_page(page)
    table_blocks = extract_specification_table_blocks(page, page_text=raw_page.raw_text)
    return clean_page_record(raw_page, table_blocks)


def iter_clean_page_records(pdf_path: str | Path) -> Iterator[PageRecord]:
    """Yield cleaned, table-aware page records in PDF order."""
    path = _validate_pdf_path(pdf_path)

    with pymupdf.open(path) as document:  # type: ignore[no-untyped-call]
        _validate_document(document, path)

        for page in document:
            yield extract_clean_page(page)


def _parse_header_block(block: TextBlock) -> tuple[str, str, str] | None:
    lines = tuple(line.strip() for line in block.text.splitlines() if line.strip())
    if len(lines) < 2:
        return None

    category = lines[-1]
    if not _CATEGORY_RE.fullmatch(category):
        return None

    heading = " ".join(lines[:-1])
    match = _SECTION_HEADING_RE.fullmatch(heading)
    if not match:
        return None

    return match.group("section_id"), match.group("section_title"), category


def _is_article_title_candidate(text: str) -> bool:
    return (
        bool(text)
        and "\n" not in text
        and len(text) <= 240
        and text not in _GENERIC_SUBHEADINGS
        and not text.startswith(_NON_TITLE_PREFIXES)
        and not _NUMBERED_ITEM_RE.match(text)
        and not text.endswith((".", ";", "?", "!"))
    )


def _find_article_title(
    blocks: tuple[TextBlock, ...],
    header_index: int,
    category: str,
) -> str | None:
    if category == "SPECIFICATIONS" or header_index + 1 >= len(blocks):
        return None

    header = blocks[header_index]
    candidate = blocks[header_index + 1]
    vertical_gap = candidate.bbox[1] - header.bbox[3]

    if vertical_gap > 80 or not _is_article_title_candidate(candidate.text):
        return None

    return candidate.text


def detect_section_context(page: PageRecord) -> SectionContext | None:
    """Return new article context when a cleaned page contains its header."""
    for index, block in enumerate(page.blocks):
        parsed_header = _parse_header_block(block)
        if not parsed_header:
            continue

        section_id, section_title, category = parsed_header
        return SectionContext(
            section_id=section_id,
            section_title=section_title,
            category=category,
            article_title=_find_article_title(page.blocks, index, category),
            start_pdf_page=page.pdf_page,
        )

    return None


def iter_contextual_pages(
    pages: Iterable[PageRecord],
) -> Iterator[ContextualPageRecord]:
    """Attach the latest article context to monotonically ordered pages."""
    current_context: SectionContext | None = None
    previous_pdf_page: int | None = None

    for page in pages:
        if previous_pdf_page is not None and page.pdf_page <= previous_pdf_page:
            raise ValueError("pages must be ordered by increasing PDF page number")

        detected_context = detect_section_context(page)
        if detected_context:
            current_context = detected_context

        yield ContextualPageRecord(page=page, section=current_context)
        previous_pdf_page = page.pdf_page
