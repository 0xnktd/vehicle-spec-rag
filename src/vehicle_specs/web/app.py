"""Streamlit presentation layer for evidence-backed specification queries."""

import logging
from time import perf_counter

import streamlit as st

from vehicle_specs.extraction import ExtractionResponse, SpecificationResult
from vehicle_specs.indexing import (
    IndexCompatibilityError,
    IndexIntegrityError,
    IndexNotFoundError,
)
from vehicle_specs.pipeline import PipelineRun
from vehicle_specs.query_service import QueryService, create_query_service
from vehicle_specs.retrieval import RetrievalHit

LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def _query_service() -> QueryService:
    """Share heavyweight local models safely across Streamlit reruns."""
    return create_query_service(rerank=True)


def _value_text(result: SpecificationResult) -> str:
    return " ".join(part for part in (result.value, result.unit) if part)


def _alternate_text(result: SpecificationResult) -> str | None:
    if not result.alternate_values:
        return None
    return ", ".join(
        f"{alternate.value} {alternate.unit}" for alternate in result.alternate_values
    )


def _render_result(result: SpecificationResult) -> None:
    with st.container(border=True):
        st.subheader(result.component)
        value_column, context_column = st.columns([2, 3])
        with value_column:
            st.metric(result.spec_type.replace("_", " ").title(), _value_text(result))
        with context_column:
            alternate = _alternate_text(result)
            if alternate:
                st.write(f"Alternate: {alternate}")
            if result.applicability:
                st.write(f"Applies to: {result.applicability}")

        st.caption(
            f"PDF page {result.source.pdf_page} · "
            f"section {result.source.section_id} · {result.source.chunk_id}"
        )
        st.code(result.source.evidence, language=None, wrap_lines=True)


def _render_status(response: ExtractionResponse) -> None:
    if response.status == "found":
        count = len(response.results)
        noun = "specification" if count == 1 else "specifications"
        st.success(f"Found {count} supported {noun}.")
    elif response.status == "ambiguous":
        st.warning(response.clarification or "The manual contains multiple matches.")
    else:
        st.info("No supported specification was found in the retrieved manual text.")


def _score_text(hit: RetrievalHit) -> str:
    parts = [f"retrieval {hit.score:.4f}"]
    if hit.rerank_score is not None:
        parts.append(f"rerank {hit.rerank_score:.4f}")
    return " · ".join(parts)


def _render_details(run: PipelineRun, elapsed_seconds: float) -> None:
    with st.expander(f"Retrieval details ({len(run.retrieved)} chunks)"):
        st.caption(
            f"Completed in {elapsed_seconds:.1f}s · model {run.response.model_name} · "
            f"pipeline {run.response.pipeline_version} · "
            f"prompt {run.response.prompt_version}"
        )
        for hit in run.retrieved:
            pages = ", ".join(str(page) for page in hit.pdf_pages)
            section = hit.section_id or "unsectioned"
            st.markdown(f"**{hit.rank}. Section {section} · PDF page {pages}**")
            st.caption(f"{hit.chunk_id} · {_score_text(hit)}")
            st.code(hit.text, language=None, wrap_lines=True)


def _error_message(error: Exception) -> str:
    if isinstance(error, IndexNotFoundError):
        return "The local index is missing. Run the ingest command before querying."
    if isinstance(error, (IndexCompatibilityError, IndexIntegrityError)):
        return "The local index is incompatible or incomplete. Rebuild it, then retry."
    return (
        "The query could not be completed. Check the model service and container logs."
    )


def _run_and_render(query: str) -> None:
    started = perf_counter()
    try:
        with st.spinner("Searching the manual and validating evidence…"):
            run = _query_service().run(query)
    except Exception as error:
        LOGGER.exception("RAG query failed")
        st.error(_error_message(error))
        st.caption(f"Failure type: {type(error).__name__}")
        return

    elapsed_seconds = perf_counter() - started
    _render_status(run.response)
    for result in run.response.results:
        _render_result(result)
    _render_details(run, elapsed_seconds)


def render_app() -> None:
    """Render the single-page local query application."""
    st.set_page_config(
        page_title="Vehicle specification search",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.title("Vehicle specification search")
    st.caption(
        "Ask for an exact value from the indexed service manual. "
        "Every returned result is checked against its cited evidence."
    )

    with st.form("query-form", border=False):
        query = st.text_input(
            "Question",
            placeholder="Torque for front brake caliper guide pin bolts",
            key="query",
        )
        submitted = st.form_submit_button(
            "Search manual",
            type="primary",
            key="submit-query",
        )

    if not submitted:
        st.caption(
            "Example: What is the torque for the front brake caliper guide pin bolts?"
        )
        return
    if not query.strip():
        st.warning("Enter a question before searching.")
        return

    _run_and_render(query.strip())
