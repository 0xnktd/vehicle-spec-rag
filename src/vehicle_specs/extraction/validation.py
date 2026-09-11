"""Deterministic validation of LLM claims against retrieved evidence."""

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vehicle_specs.retrieval import RetrievalHit

from .models import CandidateExtraction, SpecificationResult

ValidationCode = Literal[
    "chunk_not_retrieved",
    "page_mismatch",
    "section_mismatch",
    "evidence_not_found",
    "value_not_in_evidence",
    "unit_not_in_evidence",
]


class ValidationIssue(BaseModel):
    """One reason an LLM-produced result cannot be trusted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_index: int = Field(ge=0)
    chunk_id: str = Field(min_length=1)
    code: ValidationCode
    message: str = Field(min_length=1)


class EvidenceValidationError(ValueError):
    """Raised when any extracted claim lacks exact retrieved evidence."""

    def __init__(self, issues: Sequence[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        summary = "; ".join(
            f"result {issue.result_index}: {issue.code}" for issue in self.issues
        )
        super().__init__(f"extraction evidence validation failed ({summary})")


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _contains_exact_value(text: str, value: str) -> bool:
    normalized_text = _normalize_whitespace(text).casefold()
    normalized_value = _normalize_whitespace(value).casefold()
    if not normalized_value:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_value)}(?!\w)",
            normalized_text,
        )
    )


def _contains_exact_measurement(text: str, value: str, unit: str) -> bool:
    normalized_text = _normalize_whitespace(text).casefold()
    normalized_value = _normalize_whitespace(value).casefold()
    normalized_unit = _normalize_whitespace(unit).casefold()
    if not normalized_value or not normalized_unit:
        return False
    return bool(
        re.search(
            rf"(?<!\w){re.escape(normalized_value)}\s*"
            rf"{re.escape(normalized_unit)}(?!\w)",
            normalized_text,
        )
    )


def _result_issues(
    result: SpecificationResult,
    result_index: int,
    retrieved_by_id: dict[str, RetrievalHit],
) -> list[ValidationIssue]:
    source = result.source
    hit = retrieved_by_id.get(source.chunk_id)
    if hit is None:
        return [
            ValidationIssue(
                result_index=result_index,
                chunk_id=source.chunk_id,
                code="chunk_not_retrieved",
                message="cited chunk was not supplied to the extraction model",
            )
        ]

    issues: list[ValidationIssue] = []
    if source.pdf_page not in hit.pdf_pages:
        issues.append(
            ValidationIssue(
                result_index=result_index,
                chunk_id=source.chunk_id,
                code="page_mismatch",
                message=f"PDF page {source.pdf_page} is not in the cited chunk",
            )
        )
    if source.section_id != hit.section_id:
        issues.append(
            ValidationIssue(
                result_index=result_index,
                chunk_id=source.chunk_id,
                code="section_mismatch",
                message=(
                    f"section {source.section_id!r} does not match {hit.section_id!r}"
                ),
            )
        )

    normalized_evidence = _normalize_whitespace(source.evidence)
    normalized_chunk = _normalize_whitespace(hit.text)
    evidence_found = normalized_evidence.casefold() in normalized_chunk.casefold()
    if not evidence_found:
        issues.append(
            ValidationIssue(
                result_index=result_index,
                chunk_id=source.chunk_id,
                code="evidence_not_found",
                message="evidence is not a contiguous excerpt of the cited chunk",
            )
        )
        return issues

    value_with_unit = result.unit is not None and _contains_exact_measurement(
        source.evidence,
        result.value,
        result.unit,
    )
    if not _contains_exact_value(source.evidence, result.value) and not value_with_unit:
        issues.append(
            ValidationIssue(
                result_index=result_index,
                chunk_id=source.chunk_id,
                code="value_not_in_evidence",
                message=f"value {result.value!r} does not occur in the evidence",
            )
        )
    if (
        result.unit is not None
        and not _contains_exact_value(source.evidence, result.unit)
        and not value_with_unit
    ):
        issues.append(
            ValidationIssue(
                result_index=result_index,
                chunk_id=source.chunk_id,
                code="unit_not_in_evidence",
                message=f"unit {result.unit!r} does not occur in the evidence",
            )
        )

    for alternate in result.alternate_values:
        alternate_measurement = _contains_exact_measurement(
            source.evidence,
            alternate.value,
            alternate.unit,
        )
        if (
            not _contains_exact_value(source.evidence, alternate.value)
            and not alternate_measurement
        ):
            issues.append(
                ValidationIssue(
                    result_index=result_index,
                    chunk_id=source.chunk_id,
                    code="value_not_in_evidence",
                    message=(
                        f"alternate value {alternate.value!r} does not occur "
                        "in the evidence"
                    ),
                )
            )
        if (
            not _contains_exact_value(source.evidence, alternate.unit)
            and not alternate_measurement
        ):
            issues.append(
                ValidationIssue(
                    result_index=result_index,
                    chunk_id=source.chunk_id,
                    code="unit_not_in_evidence",
                    message=(
                        f"alternate unit {alternate.unit!r} does not occur "
                        "in the evidence"
                    ),
                )
            )
    return issues


def validate_candidate(
    candidate: CandidateExtraction,
    hits: Sequence[RetrievalHit],
) -> CandidateExtraction:
    """Return the candidate unchanged only when every claim is grounded."""
    retrieved_by_id = {hit.chunk_id: hit for hit in hits}
    issues = [
        issue
        for result_index, result in enumerate(candidate.results)
        for issue in _result_issues(result, result_index, retrieved_by_id)
    ]
    if issues:
        raise EvidenceValidationError(issues)
    return candidate
