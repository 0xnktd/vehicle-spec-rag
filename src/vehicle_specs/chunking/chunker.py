"""Section-aware chunking and atomic chunk persistence."""

import os
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from vehicle_specs.pdf.models import ContextualPageRecord, SectionContext

from .models import ChunkingConfig, ChunkKind, TextChunk

CHUNKER_VERSION = "1.0.0"

_TOKEN_RE = re.compile(r"\S+")
_ID_UNSAFE_RE = re.compile(r"[^a-z0-9]+")
_NUMBERED_ITEM_RE = re.compile(r"\d+[.)](?:\s|$)")
_TABLE_SEPARATOR_CELL_RE = re.compile(r":?-{3,}:?")
_STRUCTURAL_HEADING_RE = re.compile(
    r"(?:"
    r"(?:General\s+|Torque\s+)?Specifications?"
    r"|Material"
    r"|Special Tool\(s\)"
    r"|Removal(?: and Installation)?"
    r"|Installation"
    r"|Disassembly"
    r"|Assembly"
    r"|Inspection and Verification"
    r"|Principles of Operations?"
    r"|Component Tests?"
    r"|Pinpoint Tests?"
    r"|Symptom Charts?(?: — .+)?"
    r"|Normal Operation"
    r"|(?:Diagnostic Trouble Code \(DTC\)|DTC) Charts?"
    r"|Visual Inspection Charts?(?: — .+)?"
    r")",
    re.IGNORECASE,
)
_NON_HEADING_PREFIXES = (
    "- ",
    "| ",
    "CAUTION:",
    "NOTE:",
    "NOTICE:",
    "WARNING:",
    "Yes ",
    "No ",
)


@dataclass(frozen=True, slots=True)
class _SourceBlock:
    text: str
    pdf_page: int
    is_heading: bool
    starts_section: bool


@dataclass(frozen=True, slots=True)
class _DraftChunk:
    kind: ChunkKind
    blocks: tuple[_SourceBlock, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def pdf_pages(self) -> tuple[int, ...]:
        return tuple(sorted({block.pdf_page for block in self.blocks}))


def count_tokens(text: str) -> int:
    """Count deterministic whitespace-delimited lexical tokens."""
    return sum(1 for _ in _TOKEN_RE.finditer(text))


def _slice_tokens(text: str, start: int, end: int) -> str:
    matches = tuple(_TOKEN_RE.finditer(text))
    if start >= end or start >= len(matches):
        return ""

    end = min(end, len(matches))
    return text[matches[start].start() : matches[end - 1].end()].strip()


def _split_source_block(
    block: _SourceBlock,
    capacity: int,
    overlap: int,
) -> tuple[_SourceBlock, ...]:
    token_count = count_tokens(block.text)
    if token_count <= capacity:
        return (block,)

    effective_overlap = min(overlap, capacity - 1)
    step = capacity - effective_overlap
    pieces: list[_SourceBlock] = []

    for start in range(0, token_count, step):
        end = min(start + capacity, token_count)
        pieces.append(
            _SourceBlock(
                text=_slice_tokens(block.text, start, end),
                pdf_page=block.pdf_page,
                is_heading=False,
                starts_section=False,
            )
        )
        if end == token_count:
            break

    return tuple(pieces)


def _tail_blocks(
    blocks: Sequence[_SourceBlock],
    token_limit: int,
) -> tuple[_SourceBlock, ...]:
    if token_limit <= 0:
        return ()

    remaining = token_limit
    tail: list[_SourceBlock] = []

    for block in reversed(blocks):
        block_tokens = count_tokens(block.text)
        if block_tokens <= remaining:
            tail.append(block)
            remaining -= block_tokens
        else:
            text = _slice_tokens(block.text, block_tokens - remaining, block_tokens)
            if text:
                tail.append(
                    _SourceBlock(
                        text=text,
                        pdf_page=block.pdf_page,
                        is_heading=False,
                        starts_section=False,
                    )
                )
            break

        if remaining == 0:
            break

    return tuple(reversed(tail))


def _pack_prose(
    blocks: Sequence[_SourceBlock],
    capacity: int,
    overlap: int,
) -> Iterator[_DraftChunk]:
    if not blocks:
        return
    if capacity < 1:
        raise ValueError("max_tokens is too small for the section context")

    current: list[_SourceBlock] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block.text)

        if block_tokens > capacity:
            if current:
                yield _DraftChunk(kind="prose", blocks=tuple(current))
                current = []
                current_tokens = 0

            pieces = _split_source_block(block, capacity, overlap)
            for piece in pieces[:-1]:
                yield _DraftChunk(kind="prose", blocks=(piece,))

            current = [pieces[-1]]
            current_tokens = count_tokens(pieces[-1].text)
            continue

        if current and current_tokens + block_tokens > capacity:
            yield _DraftChunk(kind="prose", blocks=tuple(current))
            allowed_overlap = min(overlap, capacity - block_tokens)
            current = list(_tail_blocks(current, allowed_overlap))
            current_tokens = sum(count_tokens(item.text) for item in current)

        current.append(block)
        current_tokens += block_tokens

    if current:
        yield _DraftChunk(kind="prose", blocks=tuple(current))


def _is_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0].startswith("|") or not lines[1].startswith("|"):
        return False

    separator_cells = tuple(cell.strip() for cell in lines[1].strip("|").split("|"))
    return bool(separator_cells) and all(
        _TABLE_SEPARATOR_CELL_RE.fullmatch(cell) for cell in separator_cells
    )


def _looks_like_heading(text: str) -> bool:
    return (
        "\n" not in text
        and 0 < count_tokens(text) <= 16
        and len(text) <= 160
        and not text.startswith(_NON_HEADING_PREFIXES)
        and not _NUMBERED_ITEM_RE.match(text)
        and not text.endswith((".", ",", ";", "?", "!"))
    )


def _starts_logical_section(text: str, block_height: float) -> bool:
    return bool(_STRUCTURAL_HEADING_RE.fullmatch(text)) or (
        11.5 <= block_height <= 18 and _looks_like_heading(text)
    )


def _is_article_header(text: str, context: SectionContext | None) -> bool:
    if context is None:
        return False

    lines = tuple(line.strip() for line in text.splitlines() if line.strip())
    return (
        len(lines) >= 2
        and " ".join(lines[:-1])
        == f"SECTION {context.section_id}: {context.section_title}"
        and lines[-1] == context.category
    )


def _article_source_blocks(
    pages: Sequence[ContextualPageRecord],
) -> tuple[_SourceBlock, ...]:
    context = pages[0].section
    article_title_skipped = False
    source_blocks: list[_SourceBlock] = []

    for record in pages:
        for block in record.page.blocks:
            if _is_article_header(block.text, context):
                continue

            if (
                context
                and context.article_title
                and not article_title_skipped
                and record.page.pdf_page == context.start_pdf_page
                and block.text == context.article_title
            ):
                article_title_skipped = True
                continue

            source_blocks.append(
                _SourceBlock(
                    text=block.text,
                    pdf_page=record.page.pdf_page,
                    is_heading=_looks_like_heading(block.text),
                    starts_section=_starts_logical_section(
                        block.text,
                        block.bbox[3] - block.bbox[1],
                    ),
                )
            )

    return tuple(source_blocks)


def _context_prefix(context: SectionContext | None) -> str:
    if context is None:
        return ""

    lines = [
        f"Section {context.section_id}: {context.section_title}",
        f"Category: {context.category}",
    ]
    if context.article_title:
        lines.append(f"Article: {context.article_title}")
    return "\n".join(lines)


def _article_drafts(
    pages: Sequence[ContextualPageRecord],
    config: ChunkingConfig,
) -> Iterator[_DraftChunk]:
    context = pages[0].section
    capacity = config.max_tokens - count_tokens(_context_prefix(context))
    prose: list[_SourceBlock] = []

    def flush_prose() -> tuple[_DraftChunk, ...]:
        nonlocal prose
        drafts = tuple(_pack_prose(prose, capacity, config.overlap_tokens))
        prose = []
        return drafts

    for block in _article_source_blocks(pages):
        if _is_markdown_table(block.text):
            table_blocks: list[_SourceBlock] = []
            if prose and prose[-1].is_heading:
                table_blocks.append(prose.pop())

            yield from flush_prose()
            table_blocks.append(block)
            yield _DraftChunk(kind="table", blocks=tuple(table_blocks))
            continue

        if (
            block.starts_section
            and prose
            and any(not item.is_heading for item in prose)
        ):
            yield from flush_prose()

        prose.append(block)

    yield from flush_prose()


def _slug(value: str) -> str:
    slug = _ID_UNSAFE_RE.sub("-", value.casefold()).strip("-")
    return slug or "document"


def _chunk_id(
    source: str,
    context: SectionContext | None,
    draft: _DraftChunk,
    ordinal: int,
) -> str:
    section_id = _slug(context.section_id) if context else "unsectioned"
    article_start = context.start_pdf_page if context else draft.pdf_pages[0]
    first_page = draft.pdf_pages[0]
    last_page = draft.pdf_pages[-1]
    page_label = f"p{first_page:04d}"
    if last_page != first_page:
        page_label += f"-{last_page:04d}"

    return (
        f"{_slug(Path(source).stem)}-{section_id}-a{article_start:04d}-"
        f"{page_label}-{draft.kind}-{ordinal:03d}"
    )


def _build_article_chunks(
    pages: Sequence[ContextualPageRecord],
    source: str,
    config: ChunkingConfig,
) -> Iterator[TextChunk]:
    context = pages[0].section
    prefix = _context_prefix(context)

    for ordinal, draft in enumerate(_article_drafts(pages, config), start=1):
        text = f"{prefix}\n\n{draft.text}" if prefix else draft.text
        yield TextChunk(
            chunk_id=_chunk_id(source, context, draft, ordinal),
            source=source,
            kind=draft.kind,
            text=text,
            pdf_pages=draft.pdf_pages,
            section=context,
        )


def iter_chunks(
    pages: Iterable[ContextualPageRecord],
    *,
    source: str,
    config: ChunkingConfig | None = None,
) -> Iterator[TextChunk]:
    """Yield chunks without crossing article boundaries."""
    source = Path(source).name.strip()
    if not source:
        raise ValueError("source cannot be empty")

    config = config or ChunkingConfig()
    article_pages: list[ContextualPageRecord] = []
    current_context: SectionContext | None = None

    for record in pages:
        if article_pages and record.section != current_context:
            yield from _build_article_chunks(article_pages, source, config)
            article_pages = []

        if not article_pages:
            current_context = record.section

        article_pages.append(record)

    if article_pages:
        yield from _build_article_chunks(article_pages, source, config)


def write_chunks_jsonl(
    chunks: Iterable[TextChunk],
    output_path: str | Path,
) -> int:
    """Atomically write chunks as UTF-8 JSON Lines and return the count."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            count = 0
            for chunk in chunks:
                temporary_file.write(chunk.model_dump_json())
                temporary_file.write("\n")
                count += 1

        os.replace(temporary_path, path)
        return count
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
