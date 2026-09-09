"""Detect article boundaries and propagate their context across PDF pages."""

import re
from collections.abc import Iterable, Iterator

from .models import ContextualPageRecord, PageRecord, SectionContext, TextBlock


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
