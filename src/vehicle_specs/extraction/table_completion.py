"""Deterministic completeness guards for directly matching table rows."""

import re
from collections.abc import Sequence

from vehicle_specs.retrieval import RetrievalHit

from .evidence import ResolvedTableRow, parse_table_rows
from .models import (
    AlternateValue,
    CandidateExtraction,
    SourceEvidence,
    SpecificationResult,
)

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_NUMBER = re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?")
_AXES = frozenset({"front", "rear"})
_TORQUE_INTENT = frozenset({"tighten", "tightened", "tightening", "torque"})
_QUERY_NOISE = frozenset(
    {
        "a",
        "an",
        "does",
        "for",
        "is",
        "needed",
        "required",
        "spec",
        "specification",
        "specifications",
        "the",
        "to",
        "value",
        "what",
    }
).union(_TORQUE_INTENT)
_TORQUE_UNITS = frozenset({"lb-ft", "lb-in", "nm"})
_CAPACITY_INTENT = frozenset({"capacity", "fill", "quantity"})
_CAPACITY_NOISE = frozenset(
    {
        "a",
        "an",
        "does",
        "for",
        "how",
        "is",
        "list",
        "listed",
        "much",
        "of",
        "section",
        "the",
        "to",
        "what",
    }
).union(_CAPACITY_INTENT)
_MEASUREMENT = re.compile(
    r"(?<![\w.])(?P<value>[+-]?(?:\d[\d,]*)(?:\.\d+)?)\s*"
    r"(?P<unit>[A-Za-zµμ°%][A-Za-z0-9µμ°%./·-]*)"
)
_DIMENSION_INTENT = frozenset(
    {"clearance", "dimension", "gap", "length", "runout", "thickness"}
)
_DIMENSION_NOISE = frozenset(
    {
        "a",
        "an",
        "dimension",
        "does",
        "for",
        "is",
        "of",
        "safe",
        "specification",
        "the",
        "to",
        "what",
    }
)


def _tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _TOKEN.findall(text.casefold()):
        if len(token) > 4 and token.endswith("ies"):
            token = f"{token[:-3]}y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _hit_scope(hit: RetrievalHit) -> str:
    return " ".join(
        part for part in (hit.section_title, hit.article_title) if part is not None
    )


def _matching_rows(
    query: str,
    hits: Sequence[RetrievalHit],
) -> list[tuple[RetrievalHit, ResolvedTableRow]]:
    query_tokens = _tokens(query)
    if not query_tokens.intersection(_TORQUE_INTENT):
        return []

    anchors = query_tokens.difference(_QUERY_NOISE).difference(_AXES)
    if len(anchors) < 2:
        return []
    requested_axes = query_tokens.intersection(_AXES)

    matches: list[tuple[RetrievalHit, ResolvedTableRow, set[str]]] = []
    for hit in hits:
        if hit.kind != "table" or "torque specifications" not in hit.text.casefold():
            continue
        scope_tokens = _tokens(_hit_scope(hit))
        for row in parse_table_rows(hit.text):
            if not row.cells:
                continue
            row_tokens = _tokens(row.cells[0])
            searchable = row_tokens.union(scope_tokens)
            if requested_axes and not requested_axes.intersection(searchable):
                continue
            if anchors.issubset(searchable):
                matches.append((hit, row, row_tokens))

    row_vocabulary = set().union(*(tokens for _, _, tokens in matches))
    component_anchors = anchors.intersection(row_vocabulary)
    exact = [
        (hit, row)
        for hit, row, row_tokens in matches
        if row_tokens.difference(_AXES) == component_anchors
    ]
    if exact:
        return exact
    return [(hit, row) for hit, row, _ in matches]


def _measurement(cell: str) -> str | None:
    match = _NUMBER.search(cell)
    return match.group() if match is not None else None


def _applicability(hit: RetrievalHit, row: ResolvedTableRow) -> str | None:
    if hit.section_title is not None:
        section_axes = _tokens(hit.section_title).intersection(_AXES)
        if len(section_axes) == 1:
            return hit.section_title
    if hit.article_title is not None:
        article_axes = _tokens(hit.article_title).intersection(_AXES)
        if len(article_axes) == 1:
            return hit.article_title
    row_axes = _tokens(row.cells[0]).intersection(_AXES)
    if len(row_axes) == 1:
        return next(iter(row_axes)).title()
    return None


def _result_from_row(
    hit: RetrievalHit,
    row: ResolvedTableRow,
) -> SpecificationResult | None:
    if hit.section_id is None:
        return None

    measurements: list[AlternateValue] = []
    for index, header in enumerate(row.header_cells):
        if index >= len(row.cells) or header.casefold() not in _TORQUE_UNITS:
            continue
        value = _measurement(row.cells[index])
        if value is not None:
            measurements.append(AlternateValue(value=value, unit=header))
    if not measurements:
        return None

    primary, *alternates = measurements
    return SpecificationResult(
        component=row.cells[0],
        spec_type="torque",
        value=primary.value,
        unit=primary.unit,
        alternate_values=tuple(alternates),
        applicability=_applicability(hit, row),
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=hit.pdf_pages[0],
            section_id=hit.section_id,
            evidence=row.citation_span,
        ),
    )


def _identity(result: SpecificationResult) -> tuple[str, ...]:
    return (
        result.component.casefold(),
        result.value.casefold(),
        (result.unit or "").casefold(),
        (result.applicability or "").casefold(),
        result.source.chunk_id,
    )


def complete_torque_table_results(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
    query: str,
) -> CandidateExtraction:
    """Add every directly matching, explicitly printed torque-table row."""
    results = list(candidate.results)
    identities = {_identity(result) for result in results}
    for hit, row in _matching_rows(query, hits):
        result = _result_from_row(hit, row)
        if result is None or _identity(result) in identities:
            continue
        identities.add(_identity(result))
        results.append(result)

    if tuple(results) == candidate.results:
        return candidate
    if len(results) == 1:
        return CandidateExtraction(status="found", results=tuple(results))
    return CandidateExtraction(
        status="ambiguous",
        results=tuple(results),
        clarification=(
            candidate.clarification
            or "Which matching component or applicability do you mean?"
        ),
    )


def _matching_capacity_rows(
    query: str,
    hits: Sequence[RetrievalHit],
) -> list[tuple[RetrievalHit, ResolvedTableRow]]:
    query_tokens = _tokens(query)
    if not query_tokens.intersection(_CAPACITY_INTENT):
        return []
    anchors = query_tokens.difference(_CAPACITY_NOISE).difference(_AXES)
    if len(anchors) < 2:
        return []
    requested_axes = query_tokens.intersection(_AXES)

    matches: list[tuple[RetrievalHit, ResolvedTableRow]] = []
    for hit in hits:
        if hit.kind != "table":
            continue
        scope_tokens = _tokens(_hit_scope(hit))
        for row in parse_table_rows(hit.text):
            if not row.cells or not any(
                header.casefold() == "fill capacity" for header in row.header_cells
            ):
                continue
            searchable = _tokens(row.cells[0]).union(scope_tokens)
            if requested_axes and not requested_axes.intersection(searchable):
                continue
            if anchors.issubset(searchable):
                matches.append((hit, row))
    if len(matches) < 2:
        return matches

    def scope_score(match: tuple[RetrievalHit, ResolvedTableRow]) -> tuple[int, int]:
        scope_tokens = _tokens(_hit_scope(match[0]))
        return (
            len(query_tokens.intersection(scope_tokens)),
            -len(scope_tokens.difference(query_tokens)),
        )

    best_score = max(scope_score(match) for match in matches)
    return [match for match in matches if scope_score(match) == best_score]


def _capacity_measurements(row: ResolvedTableRow) -> tuple[AlternateValue, ...]:
    capacity_column = next(
        (
            index
            for index, header in enumerate(row.header_cells)
            if header.casefold() == "fill capacity"
        ),
        None,
    )
    if capacity_column is None:
        return ()
    if capacity_column >= len(row.cells):
        return ()

    measurements = tuple(
        AlternateValue(value=match.group("value"), unit=match.group("unit"))
        for match in _MEASUREMENT.finditer(row.cells[capacity_column])
    )
    if not measurements:
        return ()
    units = [measurement.unit.casefold() for measurement in measurements]
    if len(units) != len(set(units)):
        # Repeated units denote applicability variants, not alternate unit columns.
        return ()
    return measurements


def _capacity_result_from_row(
    hit: RetrievalHit,
    row: ResolvedTableRow,
) -> SpecificationResult | None:
    if hit.section_id is None:
        return None
    measurements = _capacity_measurements(row)
    if not measurements:
        return None
    primary, *alternates = measurements
    return SpecificationResult(
        component=row.cells[0],
        spec_type="capacity",
        value=primary.value,
        unit=primary.unit,
        alternate_values=tuple(alternates),
        applicability=_applicability(hit, row),
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=hit.pdf_pages[0],
            section_id=hit.section_id,
            evidence=row.citation_span,
        ),
    )


def complete_capacity_table_results(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
    query: str,
) -> CandidateExtraction:
    """Resolve an LLM abstention from unambiguous matching capacity cells."""
    if candidate.results:
        return candidate
    results = tuple(
        result
        for hit, row in _matching_capacity_rows(query, hits)
        if (result := _capacity_result_from_row(hit, row)) is not None
    )
    if not results:
        return candidate
    if len(results) == 1:
        return CandidateExtraction(status="found", results=results)
    return CandidateExtraction(
        status="ambiguous",
        results=results,
        clarification="Which matching capacity or applicability do you mean?",
    )


def _matching_dimension_rows(
    query: str,
    hits: Sequence[RetrievalHit],
) -> list[tuple[RetrievalHit, ResolvedTableRow]]:
    query_tokens = _tokens(query)
    if "rotor" in query_tokens:
        query_tokens.remove("rotor")
        query_tokens.add("disc")
    if not query_tokens.intersection(_DIMENSION_INTENT):
        return []
    anchors = query_tokens.difference(_DIMENSION_NOISE).difference(_AXES)
    if len(anchors) < 2:
        return []
    requested_axes = query_tokens.intersection(_AXES)

    matches: list[tuple[RetrievalHit, ResolvedTableRow]] = []
    for hit in hits:
        if hit.kind != "table":
            continue
        scope_tokens = _tokens(_hit_scope(hit))
        for row in parse_table_rows(hit.text):
            if not row.cells or not any(
                header.casefold() == "specification" for header in row.header_cells
            ):
                continue
            searchable = _tokens(row.cells[0]).union(scope_tokens)
            if requested_axes and not requested_axes.intersection(searchable):
                continue
            if anchors.issubset(searchable):
                matches.append((hit, row))
    row_vocabulary = set().union(*(_tokens(row.cells[0]) for _, row in matches))
    component_anchors = anchors.intersection(row_vocabulary)
    exact = [
        (hit, row)
        for hit, row in matches
        if _tokens(row.cells[0]).difference(_AXES) == component_anchors
    ]
    return exact or matches


def _dimension_measurements(row: ResolvedTableRow) -> tuple[AlternateValue, ...]:
    specification_column = next(
        (
            index
            for index, header in enumerate(row.header_cells)
            if header.casefold() == "specification"
        ),
        None,
    )
    if specification_column is None:
        return ()
    if specification_column >= len(row.cells):
        return ()
    measurements = tuple(
        AlternateValue(value=match.group("value"), unit=match.group("unit"))
        for match in _MEASUREMENT.finditer(row.cells[specification_column])
    )
    if not measurements:
        return ()
    units = [measurement.unit.casefold() for measurement in measurements]
    if len(units) != len(set(units)):
        return ()
    return measurements


def _dimension_result_from_row(
    hit: RetrievalHit,
    row: ResolvedTableRow,
) -> SpecificationResult | None:
    if hit.section_id is None:
        return None
    measurements = _dimension_measurements(row)
    if not measurements:
        return None
    primary, *alternates = measurements
    return SpecificationResult(
        component=row.cells[0],
        spec_type="dimension",
        value=primary.value,
        unit=primary.unit,
        alternate_values=tuple(alternates),
        applicability=_applicability(hit, row),
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=hit.pdf_pages[0],
            section_id=hit.section_id,
            evidence=row.citation_span,
        ),
    )


def complete_dimension_table_results(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
    query: str,
) -> CandidateExtraction:
    """Add directly matching, explicitly printed dimension rows for deduping."""
    results = list(candidate.results)
    identities = {_identity(result) for result in results}
    for hit, row in _matching_dimension_rows(query, hits):
        result = _dimension_result_from_row(hit, row)
        if result is None or _identity(result) in identities:
            continue
        identities.add(_identity(result))
        results.append(result)
    if tuple(results) == candidate.results:
        return candidate
    if len(results) == 1:
        return CandidateExtraction(status="found", results=tuple(results))
    return CandidateExtraction(
        status="ambiguous",
        results=tuple(results),
        clarification=(
            candidate.clarification
            or "Which matching dimension or applicability do you mean?"
        ),
    )
