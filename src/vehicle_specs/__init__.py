"""Vehicle specification extraction pipeline."""

from vehicle_specs.pdf import (
    ContextualPageRecord,
    DocumentMetadata,
    PageRecord,
    SectionContext,
    TextBlock,
    clean_block_text,
    clean_page_record,
    detect_section_context,
    extract_clean_page,
    extract_page,
    extract_specification_table_blocks,
    iter_clean_page_records,
    iter_contextual_page_records,
    iter_contextual_pages,
    iter_page_records,
    read_document_metadata,
)

__all__ = [
    "ContextualPageRecord",
    "DocumentMetadata",
    "PageRecord",
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
