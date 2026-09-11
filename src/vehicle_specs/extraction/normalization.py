"""Deterministic normalization of schema-valid extraction candidates."""

import re
from collections.abc import Sequence

from vehicle_specs.retrieval import RetrievalHit

from .evidence import ResolvedTableRow, resolve_table_row
from .models import (
    AlternateValue,
    CandidateExtraction,
    ExtractionStatus,
    SpecificationResult,
)

_IMMEDIATE_UNIT = re.compile(r"(?P<unit>(?:[A-Za-zµμ°%][A-Za-z0-9µμ°%./·-]*))")
_NUMERIC_PREFIX = re.compile(r"^\s*(?P<value>[+-]?(?:\d[\d,]*)(?:\.\d+)?)")
_WORD_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_GENERIC_FIRST_COLUMN_HEADERS = frozenset(
    {"description", "item", "part name", "specification", "value"}
)
_MEASUREMENT_TYPES = frozenset({"torque", "capacity", "dimension"})
_UNIT_HEADERS = frozenset(
    {
        "%",
        "a",
        "bar",
        "cm",
        "ft",
        "g",
        "in",
        "kg",
        "kpa",
        "l",
        "lb-ft",
        "lb-in",
        "m",
        "ml",
        "mm",
        "mpa",
        "n",
        "nm",
        "oz",
        "psi",
        "pt",
    }
)
_AXES = frozenset({"front", "rear"})
_QUERY_STOP_WORDS = frozenset(
    {"a", "an", "does", "for", "how", "is", "of", "the", "to", "what"}
)


def _has_letters(text: str) -> bool:
    return any(character.isalpha() for character in text)


def _canonical_component(
    row: ResolvedTableRow,
    value_column: int,
    current: str,
) -> str:
    if not row.cells:
        return current

    first_cell = row.cells[0]
    if value_column != 0:
        if _has_letters(first_cell):
            return first_cell
        if first_cell not in {"", "—", "-"} and row.header_cells:
            header = row.header_cells[0]
            if (
                _has_letters(header)
                and header.casefold() not in _GENERIC_FIRST_COLUMN_HEADERS
            ):
                return f"{header} {first_cell}"
        return current

    for cell in row.cells[1:]:
        if _has_letters(cell) and cell not in {"—", "-"}:
            return cell
    return current


def _immediate_unit(cell: str, value: str) -> str | None:
    normalized_cell = " ".join(cell.split())
    normalized_value = " ".join(value.split())
    match = re.search(
        rf"(?<![\w.]){re.escape(normalized_value)}\s*{_IMMEDIATE_UNIT.pattern}",
        normalized_cell,
        flags=re.IGNORECASE,
    )
    return match.group("unit") if match is not None else None


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _WORD_TOKEN.findall(text.casefold()):
        if token in _QUERY_STOP_WORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = f"{token[:-3]}y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _printed_numeric_value(cell: str, value: str) -> str | None:
    prefix = _NUMERIC_PREFIX.match(value)
    if prefix is None:
        return None
    numeric = prefix.group("value")
    match = re.search(
        rf"(?<![\w.])(?P<value>{re.escape(numeric)})(?![\d.])",
        cell,
        flags=re.IGNORECASE,
    )
    return match.group("value") if match is not None else None


def _header_unit(header: str) -> str | None:
    normalized = " ".join(header.split())
    return normalized if normalized.casefold() in _UNIT_HEADERS else None


def _canonical_measurement(
    value: str,
    unit: str,
    *,
    cell: str,
    header: str,
) -> tuple[str, str]:
    printed_value = _printed_numeric_value(cell, value)
    if printed_value is None:
        return value, unit
    canonical_unit = (
        _immediate_unit(cell, printed_value) or _header_unit(header) or unit
    )
    return printed_value, canonical_unit


def normalize_table_fields(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
    query: str | None = None,
) -> CandidateExtraction:
    """Canonicalize table component labels and immediately printed units."""
    retrieved_by_id = {hit.chunk_id: hit for hit in hits}
    normalized_results: list[SpecificationResult] = []
    changed = False

    for result in candidate.results:
        hit = retrieved_by_id.get(result.source.chunk_id)
        row = resolve_table_row(result, hit, query) if hit is not None else None
        if row is None:
            normalized_results.append(result)
            continue

        value_columns = row.value_columns(result.value, result.unit)
        if len(value_columns) != 1:
            normalized_results.append(result)
            continue

        value_column = value_columns[0]
        component = _canonical_component(row, value_column, result.component)
        value = result.value
        unit = result.unit
        alternate_values = result.alternate_values
        if (
            result.spec_type in _MEASUREMENT_TYPES
            and unit is not None
            and value_column < len(row.cells)
        ):
            header = (
                row.header_cells[value_column]
                if value_column < len(row.header_cells)
                else ""
            )
            value, unit = _canonical_measurement(
                value,
                unit,
                cell=row.cells[value_column],
                header=header,
            )

            normalized_alternates: list[AlternateValue] = []
            seen_alternates: set[tuple[str, str]] = set()
            for alternate in result.alternate_values:
                alternate_columns = row.value_columns(
                    alternate.value,
                    alternate.unit,
                )
                alternate_column = (
                    alternate_columns[0]
                    if len(alternate_columns) == 1
                    else value_column
                )
                alternate_cell = row.cells[alternate_column]
                alternate_header = (
                    row.header_cells[alternate_column]
                    if alternate_column < len(row.header_cells)
                    else ""
                )
                alternate_value, alternate_unit = _canonical_measurement(
                    alternate.value,
                    alternate.unit,
                    cell=alternate_cell,
                    header=alternate_header,
                )
                identity = (alternate_value.casefold(), alternate_unit.casefold())
                if alternate_unit.casefold() == unit.casefold():
                    continue
                if identity in seen_alternates:
                    continue
                seen_alternates.add(identity)
                normalized_alternates.append(
                    AlternateValue(value=alternate_value, unit=alternate_unit)
                )
            alternate_values = tuple(normalized_alternates)

        if (
            component == result.component
            and value == result.value
            and unit == result.unit
            and alternate_values == result.alternate_values
        ):
            normalized_results.append(result)
            continue
        normalized_results.append(
            result.model_copy(
                update={
                    "component": component,
                    "value": value,
                    "unit": unit,
                    "alternate_values": alternate_values,
                }
            )
        )
        changed = True

    if not changed:
        return candidate
    return candidate.model_copy(update={"results": tuple(normalized_results)})


def _result_axis(
    result: SpecificationResult,
    hit: RetrievalHit | None,
) -> str | None:
    text = " ".join(
        part
        for part in (
            result.component,
            result.applicability,
            hit.section_title if hit is not None else None,
            hit.article_title if hit is not None else None,
        )
        if part is not None
    )
    axes = _tokens(text).intersection(_AXES)
    return next(iter(axes)) if len(axes) == 1 else None


def _scope_score(
    result: SpecificationResult,
    hit: RetrievalHit | None,
    query_tokens: set[str],
) -> int:
    if hit is None:
        return 0
    metadata = " ".join(
        part for part in (hit.section_title, hit.article_title) if part is not None
    )
    return len(query_tokens.intersection(_tokens(metadata)))


def _prune_to_explicit_scope(
    results: Sequence[SpecificationResult],
    retrieved_by_id: dict[str, RetrievalHit],
    query: str | None,
) -> list[SpecificationResult]:
    if len(results) < 2 or query is None:
        return list(results)
    query_tokens = _tokens(query)
    scores = [
        _scope_score(
            result,
            retrieved_by_id.get(result.source.chunk_id),
            query_tokens,
        )
        for result in results
    ]
    best_score = max(scores, default=0)
    if best_score >= 2 and any(score < best_score for score in scores):
        return [
            result
            for result, score in zip(results, scores, strict=True)
            if score == best_score
        ]

    requested_axes = query_tokens.intersection(_AXES)
    if len(requested_axes) != 1:
        return list(results)
    requested_axis = next(iter(requested_axes))
    axes = [
        _result_axis(result, retrieved_by_id.get(result.source.chunk_id))
        for result in results
    ]
    if requested_axis not in axes:
        return list(results)
    opposite_axis = "rear" if requested_axis == "front" else "front"
    return [
        result
        for result, axis in zip(results, axes, strict=True)
        if axis != opposite_axis
    ]


def _equivalent_claim(
    left: SpecificationResult,
    right: SpecificationResult,
    retrieved_by_id: dict[str, RetrievalHit],
) -> bool:
    if (
        left.spec_type != right.spec_type
        or left.value.casefold() != right.value.casefold()
        or (left.unit or "").casefold() != (right.unit or "").casefold()
    ):
        return False

    left_axis = _result_axis(left, retrieved_by_id.get(left.source.chunk_id))
    right_axis = _result_axis(right, retrieved_by_id.get(right.source.chunk_id))
    if left_axis is not None and right_axis is not None and left_axis != right_axis:
        return False

    left_tokens = _tokens(left.component).difference(_AXES)
    right_tokens = _tokens(right.component).difference(_AXES)
    shared = left_tokens.intersection(right_tokens)
    if len(shared) < 2:
        return False
    if left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens):
        return True
    return len(shared) / len(left_tokens.union(right_tokens)) >= 0.75


def _representative_score(
    result: SpecificationResult,
    retrieved_by_id: dict[str, RetrievalHit],
    query_tokens: set[str],
) -> tuple[int, int, int, int, int]:
    hit = retrieved_by_id.get(result.source.chunk_id)
    if hit is None:
        return (0, 0, 0, 0, 0)
    return (
        int(hit.kind == "table"),
        _scope_score(result, hit, query_tokens),
        int(hit.category == "SPECIFICATIONS"),
        len(query_tokens.intersection(_tokens(result.component))),
        -hit.rank,
    )


def _deduplicate_results(
    results: Sequence[SpecificationResult],
    retrieved_by_id: dict[str, RetrievalHit],
    query: str | None,
) -> list[SpecificationResult]:
    groups: list[list[SpecificationResult]] = []
    for result in results:
        group = next(
            (
                existing
                for existing in groups
                if _equivalent_claim(result, existing[0], retrieved_by_id)
            ),
            None,
        )
        if group is None:
            groups.append([result])
        else:
            group.append(result)

    query_tokens = _tokens(query or "")
    return [
        max(
            group,
            key=lambda result: _representative_score(
                result,
                retrieved_by_id,
                query_tokens,
            ),
        )
        for group in groups
    ]


def _clarification(results: Sequence[SpecificationResult]) -> str:
    components = {result.component.casefold() for result in results}
    applicability = {
        result.applicability.casefold()
        for result in results
        if result.applicability is not None
    }
    if len(components) > 1 and len(applicability) > 1:
        return (
            "Which component and applicability (for example, front or rear) "
            "do you mean?"
        )
    if len(components) > 1:
        return "Which component do you mean?"
    if len(applicability) > 1:
        return "Which applicability (for example, front or rear) do you mean?"
    return "Which of these specifications do you mean?"


def normalize_candidate(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
    query: str | None = None,
) -> CandidateExtraction:
    """Deduplicate equivalent claims and normalize their response status."""
    if not candidate.results:
        return candidate

    retrieved_by_id = {hit.chunk_id: hit for hit in hits}
    deduplicated = _deduplicate_results(candidate.results, retrieved_by_id, query)
    scoped = _prune_to_explicit_scope(deduplicated, retrieved_by_id, query)

    if len(scoped) == 1:
        status: ExtractionStatus = "found"
        clarification = None
    else:
        status = "ambiguous"
        clarification = candidate.clarification or _clarification(scoped)

    normalized_results = tuple(scoped)
    if (
        normalized_results == candidate.results
        and status == candidate.status
        and clarification == candidate.clarification
    ):
        return candidate
    return CandidateExtraction(
        status=status,
        results=normalized_results,
        clarification=clarification,
    )
