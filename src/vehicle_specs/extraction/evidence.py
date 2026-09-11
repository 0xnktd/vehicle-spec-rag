"""Deterministic assembly of citation spans from retrieved evidence."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from vehicle_specs.retrieval import RetrievalHit

from .models import CandidateExtraction, SpecificationResult

_SEPARATOR_CELL = re.compile(r":?-{3,}:?")
_NUMERIC_VALUE = re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?")
_TEXT_TOKEN = re.compile(r"[^\W\d_][\w®™-]{2,}", re.UNICODE)
_EVIDENCE_STOP_WORDS = frozenset(
    {"and", "description", "for", "item", "only", "specification", "the", "with"}
)


@dataclass(frozen=True)
class ResolvedTableRow:
    """One Markdown data row together with its governing header and citation span."""

    header_cells: tuple[str, ...]
    cells: tuple[str, ...]
    row_text: str
    citation_span: str

    def value_columns(self, value: str, unit: str | None) -> tuple[int, ...]:
        """Return columns containing a claimed value, including joined measurements."""
        return tuple(
            index
            for index, cell in enumerate(self.cells)
            if _contains_claim_value(cell, value, unit)
        )


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _is_separator(line: str) -> bool:
    if not _is_table_line(line):
        return False
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return len(cells) >= 2 and all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def _parse_cells(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def parse_table_rows(text: str) -> tuple[ResolvedTableRow, ...]:
    """Parse Markdown tables into rows with contiguous citation spans."""
    lines = text.splitlines()
    rows: list[ResolvedTableRow] = []
    for separator_index, line in enumerate(lines):
        if not _is_separator(line) or separator_index == 0:
            continue
        header_index = separator_index - 1
        if not _is_table_line(lines[header_index]):
            continue
        header_cells = _parse_cells(lines[header_index])
        row_index = separator_index + 1
        while row_index < len(lines) and _is_table_line(lines[row_index]):
            if _is_separator(lines[row_index]):
                break
            rows.append(
                ResolvedTableRow(
                    header_cells=header_cells,
                    cells=_parse_cells(lines[row_index]),
                    row_text=lines[row_index],
                    citation_span="\n".join(lines[header_index : row_index + 1]),
                )
            )
            row_index += 1
    return tuple(rows)


def _contains_claim_value(text: str, value: str, unit: str | None) -> bool:
    normalized_text = _normalize(text)
    normalized_value = _normalize(value)
    if not normalized_value:
        return False

    numeric_match = _NUMERIC_VALUE.match(normalized_value) if unit is not None else None
    if numeric_match is not None:
        normalized_value = numeric_match.group()

    if unit is not None and _NUMERIC_VALUE.fullmatch(normalized_value):
        # A numeric value may be printed directly beside its unit, such as ``2.84L``.
        suffix = r"(?![\d.])"
    else:
        suffix = r"(?!\w)"
    return bool(
        re.search(
            rf"(?<![\w.]){re.escape(normalized_value)}{suffix}",
            normalized_text,
        )
    )


def _row_contains_claims(result: SpecificationResult, row: ResolvedTableRow) -> bool:
    claims = [(result.value, result.unit)] + [
        (alternate.value, alternate.unit) for alternate in result.alternate_values
    ]
    return all(
        _contains_claim_value(row.row_text, value, unit) for value, unit in claims
    )


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _TEXT_TOKEN.findall(text)
        if token.casefold() not in _EVIDENCE_STOP_WORDS
    }


def _select_row_by_anchors(
    rows: Sequence[ResolvedTableRow],
    result: SpecificationResult,
    query: str | None,
) -> ResolvedTableRow | None:
    anchors = _meaningful_tokens(
        " ".join(
            part
            for part in (result.component, result.applicability, query)
            if part is not None
        )
    )
    scored = [
        (len(anchors.intersection(_meaningful_tokens(row.row_text))), row)
        for row in rows
    ]
    best_score = max((score for score, _ in scored), default=0)
    best = [row for score, row in scored if score == best_score]
    if best_score < 2 or len(best) != 1:
        return None
    return best[0]


def resolve_table_row(
    result: SpecificationResult,
    hit: RetrievalHit,
    query: str | None = None,
) -> ResolvedTableRow | None:
    """Resolve a result to exactly one source row without fuzzy value selection."""
    if hit.kind != "table":
        return None

    table_rows = parse_table_rows(hit.text)
    rows = tuple(row for row in table_rows if _row_contains_claims(result, row))
    if not rows:
        return None

    normalized_evidence = _normalize(result.source.evidence)
    verbatim_rows = tuple(
        row
        for row in rows
        if _normalize(row.row_text) in normalized_evidence
        or normalized_evidence in _normalize(row.row_text)
    )
    if len(verbatim_rows) == 1:
        return verbatim_rows[0]
    if verbatim_rows:
        return _select_row_by_anchors(verbatim_rows, result, query)

    # Models sometimes accurately paraphrase a long table row. Repair only when
    # the claimed values select one row and the proposed evidence independently
    # shares at least two meaningful textual anchors with that row.
    if len(rows) != 1:
        return _select_row_by_anchors(rows, result, query)
    evidence_tokens = _meaningful_tokens(result.source.evidence)
    row_tokens = _meaningful_tokens(rows[0].row_text)
    if len(evidence_tokens.intersection(row_tokens)) < 2:
        return None
    return rows[0]


def assemble_table_evidence(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
    query: str | None = None,
) -> CandidateExtraction:
    """Canonicalize uniquely grounded table citations without changing claims."""
    retrieved_by_id = {hit.chunk_id: hit for hit in hits}
    updated_results: list[SpecificationResult] = []
    changed = False

    for result in candidate.results:
        hit = retrieved_by_id.get(result.source.chunk_id)
        row = resolve_table_row(result, hit, query) if hit is not None else None
        span = row.citation_span if row is not None else None
        if span is None or span == result.source.evidence:
            updated_results.append(result)
            continue

        source = result.source.model_copy(update={"evidence": span})
        updated_results.append(result.model_copy(update={"source": source}))
        changed = True

    if not changed:
        return candidate
    return candidate.model_copy(update={"results": tuple(updated_results)})
