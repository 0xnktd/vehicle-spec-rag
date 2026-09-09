"""Structure-aware chunking and chunk persistence."""

from .chunker import CHUNKER_VERSION, count_tokens, iter_chunks, iter_pdf_chunks
from .io import write_chunks_jsonl
from .models import ChunkKind, ChunkingConfig, TextChunk

__all__ = [
    "ChunkKind",
    "CHUNKER_VERSION",
    "ChunkingConfig",
    "TextChunk",
    "count_tokens",
    "iter_chunks",
    "iter_pdf_chunks",
    "write_chunks_jsonl",
]
