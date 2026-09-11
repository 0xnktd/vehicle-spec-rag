"""Structured schemas for evidence-backed vehicle specifications."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ExtractionStatus = Literal["found", "ambiguous", "not_found"]
SpecType = Literal[
    "torque",
    "capacity",
    "part_number",
    "dimension",
    "material",
    "other",
]


class AlternateValue(BaseModel):
    """An explicitly printed alternate value, normally in another unit."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    value: str = Field(
        min_length=1,
        description="Value only, exactly as printed; never include the unit.",
    )
    unit: str = Field(
        min_length=1,
        description="Unit only, exactly as printed; never include the value.",
    )


class SourceEvidence(BaseModel):
    """Exact evidence and provenance for one extracted specification."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    chunk_id: str = Field(
        min_length=1,
        description="Exact chunk_id copied from one supplied retrieved chunk.",
    )
    pdf_page: int = Field(
        ge=1,
        description="One PDF page listed for the cited retrieved chunk.",
    )
    section_id: str = Field(
        min_length=1,
        description="Exact section_id copied from the cited retrieved chunk.",
    )
    evidence: str = Field(
        min_length=1,
        description=(
            "Contiguous verbatim excerpt containing every reported value. For a "
            "table, copy the exact relevant data row; the pipeline attaches its header."
        ),
    )


class SpecificationResult(BaseModel):
    """One specification stated directly by the retrieved manual evidence."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    component: str = Field(
        min_length=1,
        description="Component or item name supported by the cited evidence.",
    )
    spec_type: SpecType
    value: str = Field(
        min_length=1,
        description=(
            "Primary value only, exactly as printed; use '37', never '37 Nm'."
        ),
    )
    unit: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Primary unit only, exactly as printed; use 'Nm', never '37 Nm'. "
            "Required for torque, capacity, and dimension results."
        ),
    )
    alternate_values: tuple[AlternateValue, ...] = Field(
        default=(),
        description=(
            "Equivalent values printed for the same item in other unit columns. "
            "Do not emit a separate result solely for another unit."
        ),
    )
    applicability: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Exact distinguishing scope, such as front/rear or engine variant, "
            "when supported by the supplied chunk metadata or text."
        ),
    )
    source: SourceEvidence

    @model_validator(mode="after")
    def validate_unit(self) -> Self:
        if self.spec_type in {"torque", "capacity", "dimension"} and self.unit is None:
            raise ValueError(f"{self.spec_type} results require a unit")
        return self


class ExtractionDraft(BaseModel):
    """Structurally valid but semantically untrusted model output."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    status: ExtractionStatus
    results: tuple[SpecificationResult, ...] = ()
    clarification: str | None = Field(default=None, min_length=1)


class CandidateExtraction(ExtractionDraft):
    """Normalized extraction candidate before deterministic evidence validation."""

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        result_count = len(self.results)
        if self.status == "not_found":
            if result_count:
                raise ValueError("not_found responses cannot contain results")
            if self.clarification is not None:
                raise ValueError("not_found responses cannot contain clarification")
        elif self.status == "found":
            if not result_count:
                raise ValueError("found responses require at least one result")
            if self.clarification is not None:
                raise ValueError("found responses cannot contain clarification")
        else:
            if result_count < 2:
                raise ValueError("ambiguous responses require at least two results")
            if self.clarification is None:
                raise ValueError("ambiguous responses require a clarification question")
        return self


class ExtractionResponse(CandidateExtraction):
    """Validated public response with reproducibility metadata."""

    model_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
