from collections.abc import Sequence

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from vehicle_specs.extraction import (
    AlternateValue,
    CandidateExtraction,
    EvidenceValidationError,
    LangChainStructuredExtractor,
    SourceEvidence,
    SpecificationResult,
    assemble_table_evidence,
    complete_capacity_table_results,
    complete_dimension_table_results,
    complete_torque_table_results,
    format_retrieval_context,
    normalize_candidate,
    normalize_table_fields,
    validate_candidate,
)
from vehicle_specs.extraction.prompt import EXTRACTION_PROMPT, SYSTEM_PROMPT
from vehicle_specs.indexing import PipelineVersions
from vehicle_specs.pipeline import VehicleSpecificationPipeline
from vehicle_specs.retrieval import RetrievalFilters, RetrievalHit


def make_hit(
    chunk_id: str = "front-torque",
    *,
    page: int = 636,
    section_id: str = "206-03",
    section_title: str = "Front Disc Brake",
    rank: int = 1,
) -> RetrievalHit:
    text = (
        f"Section {section_id}: {section_title}\n\n"
        "Torque Specifications\n"
        "| Item | Nm | lb-ft |\n"
        "| --- | --- | --- |\n"
        "| Brake caliper guide pin bolts | 37 | 27 |"
    )
    return RetrievalHit(
        point_id=rank,
        chunk_id=chunk_id,
        text=text,
        source="manual.pdf",
        kind="table",
        pdf_pages=(page,),
        pipeline_versions=PipelineVersions(),
        section_id=section_id,
        section_title=section_title,
        category="SPECIFICATIONS",
        article_title="Specifications",
        article_start_pdf_page=page,
        rank=rank,
        score=1 / rank,
    )


def make_result(
    chunk_id: str = "front-torque",
    *,
    page: int = 636,
    section_id: str = "206-03",
) -> SpecificationResult:
    return SpecificationResult(
        component="Brake caliper guide pin bolts",
        spec_type="torque",
        value="37",
        unit="Nm",
        alternate_values=(AlternateValue(value="27", unit="lb-ft"),),
        applicability="Front disc brake",
        source=SourceEvidence(
            chunk_id=chunk_id,
            pdf_page=page,
            section_id=section_id,
            evidence=(
                "| Item | Nm | lb-ft |\n"
                "| --- | --- | --- |\n"
                "| Brake caliper guide pin bolts | 37 | 27 |"
            ),
        ),
    )


def test_candidate_status_contract_is_strict() -> None:
    with pytest.raises(ValidationError, match="not_found responses cannot contain"):
        CandidateExtraction(status="not_found", results=(make_result(),))

    with pytest.raises(
        ValidationError, match="ambiguous responses require at least two"
    ):
        CandidateExtraction(
            status="ambiguous",
            results=(make_result(),),
            clarification="Which brake?",
        )


def test_context_formatter_keeps_provenance_next_to_chunk_text() -> None:
    context = format_retrieval_context([make_hit()])

    assert 'position="1"' in context
    assert "chunk_id: front-torque" in context
    assert "pdf_pages: 636" in context
    assert "section_id: 206-03" in context
    assert "Brake caliper guide pin bolts | 37 | 27" in context


def test_extraction_contract_separates_values_and_preserves_table_units() -> None:
    schema = SpecificationResult.model_json_schema()["properties"]

    assert "never '37 Nm'" in schema["value"]["description"]
    assert "Required for torque" in schema["unit"]["description"]
    assert "pipeline\n   deterministically attaches" in SYSTEM_PROMPT
    assert "copy a Markdown\n   table row exactly" in SYSTEM_PROMPT
    assert "one specification result" in SYSTEM_PROMPT
    assert "exactly the top-level keys" in SYSTEM_PROMPT
    assert '"status":"not_found"' in SYSTEM_PROMPT


def test_extraction_prompt_renders_literal_json_contract() -> None:
    rendered = EXTRACTION_PROMPT.invoke({"query": "question", "context": "evidence"})

    assert 'exactly {"status":"not_found"' in rendered.messages[0].content


def test_table_row_citation_is_deterministically_expanded_to_its_header() -> None:
    result = make_result().model_copy(
        update={
            "source": make_result().source.model_copy(
                update={"evidence": "| Brake caliper guide pin bolts | 37 | 27 |"}
            )
        }
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    assembled = assemble_table_evidence(candidate, [make_hit()])

    evidence = assembled.results[0].source.evidence
    assert evidence.startswith("| Item | Nm | lb-ft |")
    assert evidence.endswith("| Brake caliper guide pin bolts | 37 | 27 |")
    assert validate_candidate(assembled, [make_hit()]) is assembled


def test_partial_but_unique_table_row_citation_is_expanded() -> None:
    result = make_result().model_copy(
        update={
            "source": make_result().source.model_copy(
                update={"evidence": "Brake caliper guide pin bolts | 37 | 27"}
            )
        }
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    assembled = assemble_table_evidence(candidate, [make_hit()])

    assert assembled.results[0].source.evidence.startswith("| Item | Nm | lb-ft |")
    assert validate_candidate(assembled, [make_hit()]) is assembled


def test_table_evidence_assembler_does_not_repair_nonverbatim_claims() -> None:
    result = make_result().model_copy(
        update={
            "source": make_result().source.model_copy(
                update={"evidence": "invented row | 37 | 27 |"}
            )
        }
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    assert assemble_table_evidence(candidate, [make_hit()]) is candidate


def test_table_evidence_assembler_repairs_uniquely_grounded_paraphrase() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 205-02B: Rear Drive Axle\n\n"
                "| Item | Fill Capacity |\n"
                "| --- | --- |\n"
                "| SAE 75W-140 rear axle lubricant | "
                "2.84L — Non Traction-Lok equipped 2.48L (5.25 pt) |"
            ),
            "section_id": "205-02B",
            "pdf_pages": (340,),
        }
    )
    result = SpecificationResult(
        component="Rear Drive Axle/Differential",
        spec_type="capacity",
        value="2.48",
        unit="L",
        alternate_values=(AlternateValue(value="5.25", unit="pt"),),
        applicability="Non Traction-Lok equipped",
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=340,
            section_id="205-02B",
            evidence=(
                "SAE 75W-140 rear axle lubricant | 2.48L (5.25 pt) — "
                "Non Traction-Lok equipped"
            ),
        ),
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    assembled = assemble_table_evidence(candidate, [hit])

    assert assembled.results[0].source.evidence.endswith(
        "Non Traction-Lok equipped 2.48L (5.25 pt) |"
    )
    assert validate_candidate(assembled, [hit]) is assembled


def test_table_evidence_assembler_refuses_ambiguous_value_match() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 206-03: Front Disc Brake\n\n"
                "| Item | Nm |\n"
                "| --- | --- |\n"
                "| First guide pin bolt | 37 |\n"
                "| Second guide pin bolt | 37 |"
            )
        }
    )
    result = make_result().model_copy(
        update={
            "alternate_values": (),
            "source": make_result().source.model_copy(
                update={"evidence": "guide pin bolt torque is 37"}
            ),
        }
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    assert assemble_table_evidence(candidate, [hit]) is candidate


def test_table_field_normalizer_uses_source_item_and_immediate_unit() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 205-02A: Rear Drive Axle\n\n"
                "| Item | Specification | Fill Capacity |\n"
                "| --- | --- | --- |\n"
                "| Motorcraft® SAE 75W-140 Synthetic Rear Axle Lubricant | "
                "WSL-M2C192-A | 2.84L (6.0 pt) (to the filler hole) |"
            ),
            "section_id": "205-02A",
            "pdf_pages": (258,),
        }
    )
    result = SpecificationResult(
        component="Rear Axle Lubricant",
        spec_type="capacity",
        value="2.84L",
        unit="liters",
        alternate_values=(AlternateValue(value="6.0 pt", unit="pints"),),
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=258,
            section_id="205-02A",
            evidence=(
                "| Motorcraft® SAE 75W-140 Synthetic Rear Axle Lubricant | "
                "WSL-M2C192-A | 2.84L (6.0 pt) (to the filler hole) |"
            ),
        ),
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    normalized = normalize_table_fields(candidate, [hit])

    assert "75W-140" in normalized.results[0].component
    assert normalized.results[0].value == "2.84"
    assert normalized.results[0].unit == "L"
    assert normalized.results[0].alternate_values == (
        AlternateValue(value="6.0", unit="pt"),
    )
    assert validate_candidate(normalized, [hit]) is normalized


def test_table_row_resolution_uses_query_to_disambiguate_equal_values() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 206-03: Front Disc Brake\n\n"
                "| Item | mm |\n"
                "| --- | --- |\n"
                "| Brake pad taper limit | 3.0 |\n"
                "| Brake pad minimum thickness | 3.0 |"
            )
        }
    )
    result = SpecificationResult(
        component="Brake pad specification",
        spec_type="dimension",
        value="3.0",
        unit="mm",
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=636,
            section_id="206-03",
            evidence="Brake pad 3.0",
        ),
    )
    candidate = CandidateExtraction(status="found", results=(result,))
    query = "What is the brake pad minimum thickness?"

    assembled = assemble_table_evidence(candidate, [hit], query)
    normalized = normalize_table_fields(assembled, [hit], query)

    assert assembled.results[0].source.evidence.endswith(
        "| Brake pad minimum thickness | 3.0 |"
    )
    assert normalized.results[0].component == "Brake pad minimum thickness"
    assert validate_candidate(normalized, [hit]) is normalized


def test_table_field_normalizer_names_part_from_description_column() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 205-00: Driveline System\n\n"
                "| Part Number | Part Name |\n"
                "| --- | --- |\n"
                "| 00812 | Upper u-bolt center plate bolt (4 required) |"
            ),
            "section_id": "205-00",
            "pdf_pages": (222,),
        }
    )
    result = SpecificationResult(
        component="Part",
        spec_type="part_number",
        value="00812",
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=222,
            section_id="205-00",
            evidence="| 00812 | Upper u-bolt center plate bolt (4 required) |",
        ),
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    normalized = normalize_table_fields(candidate, [hit])

    assert normalized.results[0].component == (
        "Upper u-bolt center plate bolt (4 required)"
    )


def test_table_field_normalizer_drops_same_unit_applicability_variants() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 205-02B: Rear Drive Axle\n\n"
                "| Item | Fill Capacity |\n"
                "| --- | --- |\n"
                "| SAE 75W-140 axle lubricant | 2.84L (6.0 pt) — "
                "ELD only 2.60L (5.5 pt) — Non Traction-Lok equipped "
                "2.48L (5.25 pt) |"
            ),
            "section_id": "205-02B",
            "pdf_pages": (340,),
        }
    )
    result = SpecificationResult(
        component="SAE 75W-140 axle lubricant",
        spec_type="capacity",
        value="2.60",
        unit="L",
        alternate_values=(
            AlternateValue(value="2.84", unit="L"),
            AlternateValue(value="2.48", unit="L"),
        ),
        applicability="ELD only",
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=340,
            section_id="205-02B",
            evidence=(
                "| SAE 75W-140 axle lubricant | 2.84L (6.0 pt) — "
                "ELD only 2.60L (5.5 pt) — Non Traction-Lok equipped "
                "2.48L (5.25 pt) |"
            ),
        ),
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    normalized = normalize_table_fields(candidate, [hit], "ELD lubricant capacity")

    assert normalized.results[0].alternate_values == ()
    assert validate_candidate(normalized, [hit]) is normalized


def test_torque_table_completion_returns_all_generic_component_rows() -> None:
    front_hit = make_hit().model_copy(
        update={
            "text": (
                "Section 206-03: Front Disc Brake\n\n"
                "Torque Specifications\n"
                "| Description | Nm | lb-ft |\n"
                "| --- | --- | --- |\n"
                "| Brake caliper anchor plate bolts | 250 | 184 |\n"
                "| Brake caliper flow bolt | 35 | 26 |\n"
                "| Brake caliper guide pin bolts | 37 | 27 |"
            )
        }
    )
    rear_hit = make_hit(
        "rear-torque",
        page=652,
        section_id="206-04",
        section_title="Rear Disc Brake",
        rank=2,
    ).model_copy(
        update={
            "text": (
                "Section 206-04: Rear Disc Brake\n\n"
                "Torque Specifications\n"
                "| Description | Nm | lb-ft |\n"
                "| --- | --- | --- |\n"
                "| Brake caliper flow bolt | 35 | 26 |\n"
                "| Brake caliper guide pin bolts | 33 | 24 |\n"
                "| Brake caliper support bracket bolts | 150 | 111 |"
            )
        }
    )
    candidate = CandidateExtraction(status="found", results=(make_result(),))

    completed = complete_torque_table_results(
        candidate,
        [front_hit, rear_hit],
        "Torque for brake caliper bolts",
    )

    assert completed.status == "ambiguous"
    assert len(completed.results) == 6
    assert {
        (result.value, (result.applicability or "").casefold())
        for result in completed.results
    } == {
        ("250", "front disc brake"),
        ("35", "front disc brake"),
        ("37", "front disc brake"),
        ("35", "rear disc brake"),
        ("33", "rear disc brake"),
        ("150", "rear disc brake"),
    }


def test_torque_table_completion_honors_scope_and_metadata_anchors() -> None:
    abs_hit = make_hit().model_copy(
        update={
            "text": (
                "Section 206-09: Anti-Lock Brake System (ABS)\n\n"
                "Torque Specifications\n"
                "| Description | Nm | lb-in |\n"
                "| --- | --- | --- |\n"
                "| Front wheel speed sensor bolt | 17 | 150 |\n"
                "| Rear wheel speed sensor bolt | 15 | 133 |"
            ),
            "section_id": "206-09",
            "section_title": "Anti-Lock Brake System (ABS)",
            "pdf_pages": (747,),
        }
    )
    suspension_hit = make_hit("front-suspension", page=59).model_copy(
        update={
            "text": (
                "Section 204-01B: Front Suspension\n\n"
                "Torque Specifications\n"
                "| Description | Nm | lb-in |\n"
                "| --- | --- | --- |\n"
                "| Wheel speed sensor bolt | 18 | 159 |"
            ),
            "section_id": "204-01B",
            "section_title": "Front Suspension",
        }
    )

    completed = complete_torque_table_results(
        CandidateExtraction(status="not_found"),
        [abs_hit, suspension_hit],
        "ABS front wheel speed sensor bolt torque",
    )

    assert completed.status == "found"
    assert len(completed.results) == 1
    assert completed.results[0].value == "17"
    assert completed.results[0].source.pdf_page == 747


@pytest.mark.parametrize(
    ("query", "section_title", "item", "capacity", "expected_value"),
    [
        (
            "Brake system DOT 3 fluid fill capacity",
            "Brake System — General Information",
            "Motorcraft High Performance DOT 3 Motor Vehicle Brake Fluid",
            "1,420 ml (3.0 pt)",
            "1,420",
        ),
        (
            "Front drive axle 80W-90 axle lubricant capacity",
            "Front Drive Axle/Differential",
            "Motorcraft SAE 80W-90 Premium Rear Axle Lubricant",
            "1.66L (3.5 pt)",
            "1.66",
        ),
    ],
)
def test_capacity_table_completion_resolves_safe_llm_abstentions(
    query: str,
    section_title: str,
    item: str,
    capacity: str,
    expected_value: str,
) -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                f"Section 000-00: {section_title}\n\n"
                "| Item | Specification | Fill Capacity |\n"
                "| --- | --- | --- |\n"
                f"| {item} | MATERIAL-SPEC | {capacity} |"
            ),
            "section_id": "000-00",
            "section_title": section_title,
        }
    )

    completed = complete_capacity_table_results(
        CandidateExtraction(status="not_found"),
        [hit],
        query,
    )

    assert completed.status == "found"
    assert completed.results[0].value == expected_value
    assert completed.results[0].alternate_values
    if query.startswith("Front drive axle"):
        assert completed.results[0].applicability == "Front Drive Axle/Differential"
    assert validate_candidate(completed, [hit]) is completed


def test_capacity_table_completion_refuses_same_unit_applicability_variants() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 205-02B: Rear Drive Axle\n\n"
                "| Item | Specification | Fill Capacity |\n"
                "| --- | --- | --- |\n"
                "| SAE 75W-140 axle lubricant | SPEC | "
                "2.84L — ELD 2.60L — Traction-Lok 2.48L |"
            ),
            "section_id": "205-02B",
            "section_title": "Rear Drive Axle",
        }
    )
    candidate = CandidateExtraction(status="not_found")

    completed = complete_capacity_table_results(
        candidate,
        [hit],
        "rear axle lubricant fill capacity",
    )

    assert completed is candidate


def test_capacity_table_completion_prefers_closest_section_scope() -> None:
    row = (
        "| Item | Specification | Fill Capacity |\n"
        "| --- | --- | --- |\n"
        "| High Performance DOT 3 Motor Vehicle Brake Fluid | SPEC | "
    )
    general_hit = make_hit("general-fluid", page=599).model_copy(
        update={
            "text": f"Material\n{row}1,420 ml (3.0 pt) |",
            "section_id": "206-00",
            "section_title": "Brake System — General Information",
        }
    )
    abs_hit = make_hit("abs-fluid", page=747, rank=2).model_copy(
        update={
            "text": f"Material\n{row}740 ml (1.56 pt) |",
            "section_id": "206-09",
            "section_title": "Anti-Lock Brake System (ABS) and Stability Control",
        }
    )

    completed = complete_capacity_table_results(
        CandidateExtraction(status="not_found"),
        [general_hit, abs_hit],
        "Brake system DOT 3 fluid fill capacity",
    )

    assert completed.status == "found"
    assert completed.results[0].value == "1,420"
    assert completed.results[0].source.pdf_page == 599


@pytest.mark.parametrize(
    ("query", "general_item", "scoped_item", "scoped_title", "scoped_page"),
    [
        (
            "Minimum front brake disc thickness",
            "Front brake disc minimum thickness",
            "Brake disc minimum thickness",
            "Front Disc Brake",
            636,
        ),
        (
            "Minimum safe thickness of the rear brake rotor",
            "Rear brake disc minimum thickness",
            "Brake disc minimum thickness",
            "Rear Disc Brake",
            652,
        ),
        (
            "Front brake pad minimum thickness",
            "Minimum brake pad thickness",
            "Brake pad minimum thickness",
            "Front Disc Brake",
            636,
        ),
    ],
)
def test_dimension_table_completion_prefers_directly_scoped_duplicate(
    query: str,
    general_item: str,
    scoped_item: str,
    scoped_title: str,
    scoped_page: int,
) -> None:
    def dimension_hit(
        chunk_id: str,
        *,
        page: int,
        section_id: str,
        section_title: str,
        item: str,
        rank: int,
    ) -> RetrievalHit:
        return make_hit(
            chunk_id,
            page=page,
            section_id=section_id,
            section_title=section_title,
            rank=rank,
        ).model_copy(
            update={
                "text": (
                    "General Specifications\n"
                    "| Item | Specification |\n"
                    "| --- | --- |\n"
                    f"| {item} | 3.0 mm (0.118 in) |"
                )
            }
        )

    general_hit = dimension_hit(
        "general-dimension",
        page=599,
        section_id="206-00",
        section_title="Brake System — General Information",
        item=general_item,
        rank=1,
    )
    scoped_hit = dimension_hit(
        "scoped-dimension",
        page=scoped_page,
        section_id="206-03" if scoped_page == 636 else "206-04",
        section_title=scoped_title,
        item=scoped_item,
        rank=2,
    )
    model_result = SpecificationResult(
        component=general_item,
        spec_type="dimension",
        value="3.0",
        unit="mm",
        alternate_values=(AlternateValue(value="0.118", unit="in"),),
        applicability="Front" if "front" in query.casefold() else "Rear",
        source=SourceEvidence(
            chunk_id=general_hit.chunk_id,
            pdf_page=599,
            section_id="206-00",
            evidence=f"| {general_item} | 3.0 mm (0.118 in) |",
        ),
    )
    candidate = CandidateExtraction(status="found", results=(model_result,))

    completed = complete_dimension_table_results(
        candidate,
        [general_hit, scoped_hit],
        query,
    )
    normalized = normalize_candidate(completed, [general_hit, scoped_hit], query)

    assert normalized.status == "found"
    assert normalized.results[0].source.pdf_page == scoped_page


def test_dimension_table_completion_does_not_confuse_section_and_row_terms() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 206-03: Front Disc Brake\n\n"
                "General Specifications\n"
                "| Item | Specification |\n"
                "| --- | --- |\n"
                "| Brake disc minimum thickness | 32 mm (1.259 in) |\n"
                "| Brake pad minimum thickness | 3.0 mm (0.118 in) |"
            )
        }
    )

    completed = complete_dimension_table_results(
        CandidateExtraction(status="not_found"),
        [hit],
        "Minimum front brake disc thickness",
    )

    assert completed.status == "found"
    assert len(completed.results) == 1
    assert completed.results[0].component == "Brake disc minimum thickness"


def test_candidate_normalizer_deduplicates_same_claim_across_sources() -> None:
    duplicate_hit = make_hit("front-procedure", rank=2)
    duplicate = make_result("front-procedure")
    candidate = CandidateExtraction(
        status="found",
        results=(make_result(), duplicate),
    )

    normalized = normalize_candidate(candidate, [make_hit(), duplicate_hit])

    assert normalized.status == "found"
    assert len(normalized.results) == 1
    assert normalized.results[0].source.chunk_id == "front-torque"


def test_candidate_normalizer_marks_distinct_claims_ambiguous() -> None:
    rear_hit = make_hit(
        "rear-torque",
        page=652,
        section_id="206-04",
        section_title="Rear Disc Brake",
        rank=2,
    )
    rear = make_result(
        "rear-torque",
        page=652,
        section_id="206-04",
    ).model_copy(update={"value": "33", "applicability": "Rear Disc Brake"})
    front = make_result().model_copy(update={"applicability": "Front Disc Brake"})
    candidate = CandidateExtraction(status="found", results=(front, rear))

    normalized = normalize_candidate(candidate, [make_hit(), rear_hit])

    assert normalized.status == "ambiguous"
    assert len(normalized.results) == 2
    assert normalized.clarification is not None
    assert "front or rear" in normalized.clarification


def test_candidate_normalizer_applies_explicit_front_scope() -> None:
    rear_hit = make_hit(
        "rear-torque",
        page=652,
        section_id="206-04",
        section_title="Rear Disc Brake",
        rank=2,
    )
    rear = make_result(
        "rear-torque",
        page=652,
        section_id="206-04",
    ).model_copy(update={"value": "33", "applicability": "Rear Disc Brake"})
    front = make_result().model_copy(update={"applicability": "Front Disc Brake"})
    candidate = CandidateExtraction(
        status="ambiguous",
        results=(front, rear),
        clarification="Front or rear?",
    )

    normalized = normalize_candidate(
        candidate,
        [make_hit(), rear_hit],
        "front guide pin torque",
    )

    assert normalized.status == "found"
    assert normalized.clarification is None
    assert normalized.results == (front,)


def test_candidate_normalizer_prefers_directly_scoped_section() -> None:
    general_hit = make_hit(
        "general-fluid",
        page=599,
        section_id="206-00",
        section_title="Brake System General Information",
        rank=1,
    )
    front_hit = make_hit(
        "front-fluid",
        page=636,
        section_id="206-03",
        section_title="Front Disc Brake",
        rank=2,
    )
    general = make_result(
        "general-fluid",
        page=599,
        section_id="206-00",
    ).model_copy(update={"component": "Brake fluid fill capacity"})
    front = make_result(
        "front-fluid",
        page=636,
        section_id="206-03",
    ).model_copy(update={"component": "Brake fluid fill capacity"})
    candidate = CandidateExtraction(
        status="ambiguous",
        results=(general, front),
        clarification="Which section?",
    )

    normalized = normalize_candidate(
        candidate,
        [general_hit, front_hit],
        "How much brake fluid does the front disc brake section list?",
    )

    assert normalized.status == "found"
    assert normalized.results == (front,)


def test_candidate_normalizer_prefers_table_for_corroborating_claim() -> None:
    table_hit = make_hit(rank=2)
    prose_hit = make_hit("front-procedure", rank=1).model_copy(
        update={"kind": "text", "category": "REMOVAL AND INSTALLATION"}
    )
    table_result = make_result()
    prose_result = make_result("front-procedure")
    candidate = CandidateExtraction(
        status="found",
        results=(prose_result, table_result),
    )

    normalized = normalize_candidate(
        candidate,
        [prose_hit, table_hit],
        "brake caliper guide pin bolt torque",
    )

    assert normalized.status == "found"
    assert normalized.results == (table_result,)


def test_candidate_normalizer_keeps_distinct_subtypes_with_same_value() -> None:
    guide_pin = make_result().model_copy(update={"value": "35", "alternate_values": ()})
    flow_bolt = make_result().model_copy(
        update={
            "component": "Brake caliper flow bolt",
            "value": "35",
            "alternate_values": (),
        }
    )
    candidate = CandidateExtraction(
        status="ambiguous",
        results=(guide_pin, flow_bolt),
        clarification="Which bolt?",
    )

    normalized = normalize_candidate(
        candidate,
        [make_hit()],
        "brake caliper bolt torque",
    )

    assert normalized.status == "ambiguous"
    assert len(normalized.results) == 2


def test_langchain_extractor_validates_structured_chain_output() -> None:
    captured: list[dict[str, str]] = []

    def invoke(inputs: dict[str, str]) -> dict:
        captured.append(inputs)
        return CandidateExtraction(
            status="found",
            results=(make_result(),),
        ).model_dump(mode="json")

    extractor = LangChainStructuredExtractor(
        model_name="fake-local-model",
        _chain=RunnableLambda(invoke),
    )

    candidate = extractor.extract("front guide pin torque", [make_hit()])

    assert candidate.status == "found"
    assert candidate.results[0].value == "37"
    assert captured[0]["query"] == "front guide pin torque"
    assert "front-torque" in captured[0]["context"]


def test_langchain_extractor_repairs_inconsistent_model_status() -> None:
    def invoke(inputs: dict[str, str]) -> dict:
        return {
            "status": "ambiguous",
            "results": [make_result().model_dump(mode="json")],
            "clarification": "Front or rear?",
        }

    extractor = LangChainStructuredExtractor(
        model_name="fake-local-model",
        _chain=RunnableLambda(invoke),
    )

    candidate = extractor.extract("front guide pin torque", [make_hit()])

    assert candidate.status == "found"
    assert candidate.results == (make_result(),)
    assert candidate.clarification is None


def test_extractor_skips_llm_when_retrieval_is_empty() -> None:
    calls = 0

    def invoke(inputs: dict[str, str]) -> dict:
        nonlocal calls
        calls += 1
        return {"status": "not_found", "results": [], "clarification": None}

    extractor = LangChainStructuredExtractor(
        model_name="fake-local-model",
        _chain=RunnableLambda(invoke),
    )

    assert extractor.extract("unknown specification", []).status == "not_found"
    assert calls == 0


def test_evidence_validator_accepts_exact_grounded_result() -> None:
    candidate = CandidateExtraction(status="found", results=(make_result(),))

    assert validate_candidate(candidate, [make_hit()]) is candidate


def test_evidence_validator_accepts_a_value_joined_to_its_unit() -> None:
    hit = make_hit().model_copy(
        update={
            "text": (
                "Section 205-02A: Rear Drive Axle\n\n"
                "| Item | Fill Capacity |\n"
                "| --- | --- |\n"
                "| SAE 75W-140 axle lubricant | 2.84L (6.0 pt) |"
            ),
            "section_id": "205-02A",
            "pdf_pages": (258,),
        }
    )
    result = SpecificationResult(
        component="SAE 75W-140 axle lubricant",
        spec_type="capacity",
        value="2.84",
        unit="L",
        alternate_values=(AlternateValue(value="6.0", unit="pt"),),
        source=SourceEvidence(
            chunk_id=hit.chunk_id,
            pdf_page=258,
            section_id="205-02A",
            evidence="| SAE 75W-140 axle lubricant | 2.84L (6.0 pt) |",
        ),
    )
    candidate = CandidateExtraction(status="found", results=(result,))

    assert validate_candidate(candidate, [hit]) is candidate


@pytest.mark.parametrize(
    ("updated_source", "code"),
    [
        ({"chunk_id": "not-retrieved"}, "chunk_not_retrieved"),
        ({"pdf_page": 999}, "page_mismatch"),
        ({"section_id": "999-99"}, "section_mismatch"),
        ({"evidence": "invented evidence 37 Nm"}, "evidence_not_found"),
        (
            {
                "evidence": (
                    "| Item | Nm | lb-ft |\n"
                    "| --- | --- | --- |\n"
                    "| Brake caliper guide pin bolts | 37 | 27 |"
                )
            },
            "value_not_in_evidence",
        ),
    ],
)
def test_evidence_validator_rejects_unsupported_claims(
    updated_source: dict,
    code: str,
) -> None:
    result = make_result()
    if code == "value_not_in_evidence":
        result = result.model_copy(update={"value": "99"})
    else:
        source = result.source.model_copy(update=updated_source)
        result = result.model_copy(update={"source": source})
    candidate = CandidateExtraction(status="found", results=(result,))

    with pytest.raises(EvidenceValidationError) as raised:
        validate_candidate(candidate, [make_hit()])

    assert code in {issue.code for issue in raised.value.issues}


class FakeRetriever:
    def __init__(self, hits: Sequence[RetrievalHit]) -> None:
        self.hits = list(hits)

    def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        assert query == "front guide pin torque"
        assert filters is None
        return self.hits


class FakeExtractor:
    model_name = "fake-local-model"
    prompt_version = "prompt-test"

    def extract(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
    ) -> CandidateExtraction:
        assert query == "front guide pin torque"
        assert len(hits) == 1
        return CandidateExtraction(status="found", results=(make_result(),))


def test_complete_pipeline_returns_only_validated_claims() -> None:
    pipeline = VehicleSpecificationPipeline(
        FakeRetriever([make_hit()]),
        FakeExtractor(),
    )

    run = pipeline.run("  front guide pin torque  ")

    assert run.query == "front guide pin torque"
    assert run.response.status == "found"
    assert run.response.model_name == "fake-local-model"
    assert run.response.prompt_version == "prompt-test"
    assert run.response.pipeline_version == "1.5.0"
    assert run.retrieved[0].chunk_id == "front-torque"
