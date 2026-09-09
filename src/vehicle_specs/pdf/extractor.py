"""Open PDF documents and stream raw or normalized page records."""

from collections.abc import Iterator
from pathlib import Path

import pymupdf

from .cleaner import clean_page_record
from .models import ContextualPageRecord, DocumentMetadata, PageRecord, TextBlock
from .sections import iter_contextual_pages
from .tables import extract_specification_table_blocks


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


def read_document_metadata(pdf_path: str | Path) -> DocumentMetadata:
    """Read the minimal metadata needed by the extraction pipeline."""
    path = _validate_pdf_path(pdf_path)

    with pymupdf.open(path) as document:
        _validate_document(document, path)
        return DocumentMetadata(
            filename=path.name,
            page_count=document.page_count,
        )


def extract_page(page: pymupdf.Page) -> PageRecord:
    """Extract ordered text blocks and their coordinates from one PDF page."""
    text_blocks: list[TextBlock] = []

    for block in page.get_text("blocks", sort=True):
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


def iter_page_records(pdf_path: str | Path) -> Iterator[PageRecord]:
    """Yield page records in PDF order without retaining the entire document."""
    path = _validate_pdf_path(pdf_path)

    with pymupdf.open(path) as document:
        _validate_document(document, path)

        for page in document:
            yield extract_page(page)


def iter_clean_page_records(pdf_path: str | Path) -> Iterator[PageRecord]:
    """Yield cleaned, table-aware page records in PDF order."""
    path = _validate_pdf_path(pdf_path)

    with pymupdf.open(path) as document:
        _validate_document(document, path)

        for page in document:
            yield extract_clean_page(page)


def iter_contextual_page_records(pdf_path: str | Path) -> Iterator[ContextualPageRecord]:
    """Yield cleaned pages with article context propagated across page boundaries."""
    yield from iter_contextual_pages(iter_clean_page_records(pdf_path))
