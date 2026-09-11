"""Atomic persistence of cleaned page records for inspection."""

import os
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import PageRecord
from .quality import PageQuality


def write_page_records_jsonl(
    pages: Iterable[PageRecord],
    output_path: str | Path,
) -> int:
    """Atomically write cleaned page records as UTF-8 JSON Lines."""
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
            for page in pages:
                temporary_file.write(page.model_dump_json())
                temporary_file.write("\n")
                count += 1

        os.replace(temporary_path, path)
        return count
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def write_page_quality_jsonl(
    qualities: Iterable[PageQuality],
    output_path: str | Path,
) -> int:
    """Atomically write page-quality measurements as UTF-8 JSON Lines."""
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
            for quality in qualities:
                temporary_file.write(quality.model_dump_json())
                temporary_file.write("\n")
                count += 1

        os.replace(temporary_path, path)
        return count
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
