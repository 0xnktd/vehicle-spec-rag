"""PDF extraction, cleanup, table reconstruction, and structural context."""

from .cleaner import PDF_CLEANER_VERSION, clean_block_text, clean_page_record
from .extractor import (
    PDF_PARSER_VERSION,
    extract_clean_page,
    extract_page,
    iter_clean_page_records,
    iter_contextual_page_records,
    iter_page_records,
    read_document_metadata,
)
from .models import (
    ContextualPageRecord,
    DocumentMetadata,
    PageRecord,
    SectionContext,
    TextBlock,
)
from .sections import detect_section_context, iter_contextual_pages
from .tables import extract_specification_table_blocks

__all__ = [
    "ContextualPageRecord",
    "DocumentMetadata",
    "PageRecord",
    "PDF_CLEANER_VERSION",
    "PDF_PARSER_VERSION",
    "SectionContext",
    "TextBlock",
    "clean_block_text",
    "clean_page_record",
    "detect_section_context",
    "extract_clean_page",
    "extract_page",
    "extract_specification_table_blocks",
    "iter_clean_page_records",
    "iter_contextual_page_records",
    "iter_contextual_pages",
    "iter_page_records",
    "read_document_metadata",
]
