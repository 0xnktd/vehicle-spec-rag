"""Persistence helpers for inspecting and indexing generated chunks."""

import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import TextChunk


def write_chunks_jsonl(chunks: Iterable[TextChunk], output_path: str | Path) -> int:
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
