"""PDF extraction, cleanup, table reconstruction, and structural context."""

from .cleaner import PDF_CLEANER_VERSION, clean_block_text, clean_page_record
from .extractor import (
    PDF_PARSER_VERSION,
    detect_section_context,
    extract_clean_page,
    extract_page,
    iter_clean_page_records,
    iter_contextual_pages,
)
from .io import write_page_quality_jsonl, write_page_records_jsonl
from .models import (
    ContextualPageRecord,
    PageRecord,
    SectionContext,
    TextBlock,
)
from .quality import PageQuality, assess_page_quality, assess_pages
from .tables import extract_specification_table_blocks

__all__ = [
    "PDF_CLEANER_VERSION",
    "PDF_PARSER_VERSION",
    "ContextualPageRecord",
    "PageQuality",
    "PageRecord",
    "SectionContext",
    "TextBlock",
    "assess_page_quality",
    "assess_pages",
    "clean_block_text",
    "clean_page_record",
    "detect_section_context",
    "extract_clean_page",
    "extract_page",
    "extract_specification_table_blocks",
    "iter_clean_page_records",
    "iter_contextual_pages",
    "write_page_quality_jsonl",
    "write_page_records_jsonl",
]
