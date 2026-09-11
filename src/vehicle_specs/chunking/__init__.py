"""Structure-aware chunking and chunk persistence."""

from .chunker import CHUNKER_VERSION, count_tokens, iter_chunks, write_chunks_jsonl
from .models import ChunkingConfig, ChunkKind, TextChunk

__all__ = [
    "CHUNKER_VERSION",
    "ChunkKind",
    "ChunkingConfig",
    "TextChunk",
    "count_tokens",
    "iter_chunks",
    "write_chunks_jsonl",
]
