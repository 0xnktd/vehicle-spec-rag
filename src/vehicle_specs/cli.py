"""Command-line interface for the local vehicle-specification pipeline."""

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from vehicle_specs.config import AppSettings
from vehicle_specs.evaluation import (
    EvaluationExecutionError,
    EvaluationRegressionError,
    enforce_no_evaluation_failures,
    enforce_retrieval_thresholds,
    evaluate,
    load_gold_queries,
    write_evaluation_report,
)
from vehicle_specs.extraction import (
    ExtractionResponse,
    create_openai_compatible_extractor,
)
from vehicle_specs.indexing import open_index, read_index_manifest
from vehicle_specs.ingestion import ingest_pdf
from vehicle_specs.pipeline import PipelineRun, VehicleSpecificationPipeline
from vehicle_specs.query_service import create_query_service
from vehicle_specs.retrieval import (
    FastEmbedReranker,
    RetrievalFilters,
    RetrievalHit,
    VehicleSpecRetriever,
)

app = typer.Typer(
    name="vehicle-specs",
    help="Extract evidence-backed vehicle specifications from a local service manual.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


class RetrievalModeOption(StrEnum):
    """Retrieval strategies exposed by the CLI."""

    dense = "dense"
    sparse = "sparse"
    hybrid = "hybrid"


class ApplicabilityOption(StrEnum):
    """Reliable front/rear metadata filters."""

    front = "front"
    rear = "rear"


class OutputFormat(StrEnum):
    """Supported query output representations."""

    json = "json"
    human = "human"


def _echo_model(model: BaseModel) -> None:
    typer.echo(model.model_dump_json(indent=2))


def _echo_hits(hits: list[RetrievalHit]) -> None:
    typer.echo(
        json.dumps(
            [hit.model_dump(mode="json") for hit in hits],
            indent=2,
            ensure_ascii=False,
        )
    )


def _build_filters(
    *,
    source: str | None,
    sections: list[str] | None,
    categories: list[str] | None,
    applicability: ApplicabilityOption | None,
) -> RetrievalFilters | None:
    if not any((source, sections, categories, applicability)):
        return None
    return RetrievalFilters(
        source=source,
        section_ids=tuple(sections or ()),
        categories=tuple(categories or ()),
        applicability=applicability.value if applicability else None,
    )


def _create_reranker(
    settings: AppSettings,
    *,
    enabled: bool,
) -> FastEmbedReranker | None:
    if not enabled:
        return None
    return FastEmbedReranker(
        model_name=settings.reranker_model,
        batch_size=settings.reranker_batch_size,
        cache_dir=str(settings.model_cache_path),
    )


def _human_response(response: ExtractionResponse) -> str:
    lines = [f"Status: {response.status}"]
    for index, result in enumerate(response.results, start=1):
        value = f"{result.value} {result.unit or ''}".rstrip()
        lines.extend(
            [
                "",
                f"{index}. {result.component}",
                f"   {result.spec_type}: {value}",
            ]
        )
        if result.alternate_values:
            alternatives = ", ".join(
                f"{alternate.value} {alternate.unit}"
                for alternate in result.alternate_values
            )
            lines.append(f"   alternate values: {alternatives}")
        if result.applicability:
            lines.append(f"   applicability: {result.applicability}")
        lines.extend(
            [
                (
                    f"   source: {result.source.chunk_id}, PDF page "
                    f"{result.source.pdf_page}, section {result.source.section_id}"
                ),
                f"   evidence: {result.source.evidence}",
            ]
        )
    if response.clarification:
        lines.extend(["", f"Clarification: {response.clarification}"])
    lines.extend(
        [
            "",
            (
                f"Pipeline: {response.pipeline_version}; prompt: "
                f"{response.prompt_version}; model: {response.model_name}"
            ),
        ]
    )
    return "\n".join(lines)


def _human_debug(run: PipelineRun) -> str:
    lines = [_human_response(run.response), "", "Retrieved chunks:"]
    for hit in run.retrieved:
        score_parts = [f"score={hit.score:.6f}"]
        if hit.rerank_score is not None:
            score_parts.append(f"rerank={hit.rerank_score:.6f}")
        lines.extend(
            [
                "",
                f"[{hit.rank}] {hit.chunk_id} ({', '.join(score_parts)})",
                hit.text,
            ]
        )
    return "\n".join(lines)


@app.command()
def ingest(
    pdf: Annotated[
        Path | None,
        typer.Option("--pdf", help="Text-based service-manual PDF."),
    ] = None,
    index_path: Annotated[
        Path | None,
        typer.Option("--index-path", help="Persistent local Qdrant directory."),
    ] = None,
    pages_path: Annotated[
        Path | None,
        typer.Option("--pages-path", help="Cleaned-page JSONL artifact."),
    ] = None,
    chunks_path: Annotated[
        Path | None,
        typer.Option("--chunks-path", help="Retrieval-chunk JSONL artifact."),
    ] = None,
    quality_path: Annotated[
        Path | None,
        typer.Option("--quality-path", help="Page-quality JSONL artifact."),
    ] = None,
    rebuild: Annotated[
        bool,
        typer.Option("--rebuild", help="Explicitly replace an existing collection."),
    ] = False,
) -> None:
    """Extract, clean, chunk, embed, and persistently index a PDF."""
    settings = AppSettings()
    result = ingest_pdf(
        pdf or settings.pdf_path,
        index_config=settings.index_config(path=index_path),
        chunking_config=settings.chunking_config(),
        pages_path=pages_path or settings.pages_path,
        chunks_path=chunks_path or settings.chunks_path,
        page_quality_path=quality_path or settings.page_quality_path,
        rebuild=rebuild,
    )
    _echo_model(result)


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="Natural-language specification query.")],
    mode: Annotated[
        RetrievalModeOption | None,
        typer.Option("--mode", help="Dense, sparse, or reciprocal-rank hybrid."),
    ] = None,
    candidate_k: Annotated[
        int | None,
        typer.Option("--candidate-k", min=1, help="Candidates per retrieval branch."),
    ] = None,
    context_k: Annotated[
        int | None,
        typer.Option("--context-k", min=1, help="Chunks returned after reranking."),
    ] = None,
    rerank: Annotated[
        bool,
        typer.Option("--rerank", help="Apply the configured local cross-encoder."),
    ] = False,
    source: Annotated[
        str | None,
        typer.Option("--source", help="Require an exact indexed source filename."),
    ] = None,
    section: Annotated[
        list[str] | None,
        typer.Option("--section", help="Allowed section ID; repeat as needed."),
    ] = None,
    category: Annotated[
        list[str] | None,
        typer.Option("--category", help="Allowed article category; repeat as needed."),
    ] = None,
    applicability: Annotated[
        ApplicabilityOption | None,
        typer.Option("--applicability", help="Restrict to front or rear sections."),
    ] = None,
    index_path: Annotated[
        Path | None,
        typer.Option("--index-path", help="Persistent local Qdrant directory."),
    ] = None,
) -> None:
    """Retrieve ranked evidence without invoking an LLM."""
    settings = AppSettings()
    retrieval_config = settings.retrieval_config(
        mode=mode.value if mode else None,
        candidate_k=candidate_k,
        context_k=context_k,
    )
    filters = _build_filters(
        source=source,
        sections=section,
        categories=category,
        applicability=applicability,
    )
    with open_index(settings.index_config(path=index_path)) as store:
        retriever = VehicleSpecRetriever(
            store,
            retrieval_config,
            reranker=_create_reranker(settings, enabled=rerank),
        )
        _echo_hits(retriever.retrieve(query, filters=filters))


@app.command()
def query(
    query_text: Annotated[
        str,
        typer.Argument(metavar="QUERY", help="Natural-language specification query."),
    ],
    mode: Annotated[
        RetrievalModeOption | None,
        typer.Option("--mode", help="Dense, sparse, or reciprocal-rank hybrid."),
    ] = None,
    candidate_k: Annotated[
        int | None,
        typer.Option("--candidate-k", min=1, help="Candidates per retrieval branch."),
    ] = None,
    context_k: Annotated[
        int | None,
        typer.Option("--context-k", min=1, help="Chunks passed to the extractor."),
    ] = None,
    rerank: Annotated[
        bool,
        typer.Option("--rerank", help="Apply the configured local cross-encoder."),
    ] = False,
    section: Annotated[
        list[str] | None,
        typer.Option("--section", help="Allowed section ID; repeat as needed."),
    ] = None,
    applicability: Annotated[
        ApplicabilityOption | None,
        typer.Option("--applicability", help="Restrict to front or rear sections."),
    ] = None,
    output: Annotated[
        OutputFormat,
        typer.Option("--output", help="Machine-readable JSON or concise text."),
    ] = OutputFormat.json,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Include retrieved chunks and ranking scores."),
    ] = False,
    index_path: Annotated[
        Path | None,
        typer.Option("--index-path", help="Persistent local Qdrant directory."),
    ] = None,
) -> None:
    """Retrieve evidence and extract through an OpenAI-compatible model server."""
    settings = AppSettings()
    retrieval_config = settings.retrieval_config(
        mode=mode.value if mode else None,
        candidate_k=candidate_k,
        context_k=context_k or settings.extraction_context_k,
    )
    filters = _build_filters(
        source=None,
        sections=section,
        categories=None,
        applicability=applicability,
    )
    run = create_query_service(
        settings,
        index_path=index_path,
        retrieval_config=retrieval_config,
        rerank=rerank,
    ).run(query_text, filters=filters)

    if output == OutputFormat.human:
        typer.echo(_human_debug(run) if debug else _human_response(run.response))
    elif debug:
        _echo_model(run)
    else:
        _echo_model(run.response)


@app.command("evaluate")
def evaluate_command(
    gold_path: Annotated[
        Path,
        typer.Option("--gold-path", help="Manually verified JSONL gold cases."),
    ] = Path("eval/gold_queries.jsonl"),
    json_report: Annotated[
        Path,
        typer.Option("--json-report", help="Machine-readable report destination."),
    ] = Path("artifacts/evaluation.json"),
    markdown_report: Annotated[
        Path,
        typer.Option("--markdown-report", help="Readable report destination."),
    ] = Path("artifacts/evaluation.md"),
    mode: Annotated[
        RetrievalModeOption,
        typer.Option("--mode", help="Retrieval configuration to evaluate."),
    ] = RetrievalModeOption.hybrid,
    rerank: Annotated[
        bool,
        typer.Option("--rerank", help="Evaluate the configured cross-encoder."),
    ] = False,
    with_llm: Annotated[
        bool,
        typer.Option(
            "--with-llm",
            help="Also evaluate the OpenAI-compatible extraction server.",
        ),
    ] = False,
    minimum_recall_at_5: Annotated[
        float,
        typer.Option("--min-recall-at-5", min=0, max=1),
    ] = 0.70,
    minimum_recall_at_10: Annotated[
        float,
        typer.Option("--min-recall-at-10", min=0, max=1),
    ] = 0.80,
    index_path: Annotated[
        Path | None,
        typer.Option("--index-path", help="Persistent local Qdrant directory."),
    ] = None,
) -> None:
    """Measure retrieval and optionally extraction against verified cases."""
    settings = AppSettings()
    candidate_k = settings.retrieval_candidate_k
    if candidate_k < 10:
        raise typer.BadParameter(
            "evaluation requires retrieval_candidate_k of at least 10"
        )
    evaluation_retrieval_config = settings.retrieval_config(
        mode=mode.value,
        candidate_k=candidate_k,
        context_k=max(10, settings.retrieval_context_k),
    )
    gold_queries = load_gold_queries(gold_path)

    with open_index(settings.index_config(path=index_path)) as store:
        reranker = _create_reranker(settings, enabled=rerank)
        retriever = VehicleSpecRetriever(
            store,
            evaluation_retrieval_config,
            reranker=reranker,
        )
        pipeline = None
        if with_llm:
            extraction_retriever = VehicleSpecRetriever(
                store,
                settings.retrieval_config(
                    mode=mode.value,
                    candidate_k=candidate_k,
                    context_k=settings.extraction_context_k,
                ),
                reranker=reranker,
            )
            extractor = create_openai_compatible_extractor(
                settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                max_output_tokens=settings.llm_max_output_tokens,
                request_timeout_seconds=settings.llm_request_timeout_seconds,
            )
            pipeline = VehicleSpecificationPipeline(extraction_retriever, extractor)
        report = evaluate(gold_queries, retriever, pipeline=pipeline)

    write_evaluation_report(
        report,
        json_path=json_report,
        markdown_path=markdown_report,
    )
    _echo_model(report)
    try:
        enforce_no_evaluation_failures(report)
        enforce_retrieval_thresholds(
            report,
            minimum_recall_at_5=minimum_recall_at_5,
            minimum_recall_at_10=minimum_recall_at_10,
        )
    except (EvaluationExecutionError, EvaluationRegressionError) as error:
        typer.echo(f"evaluation failed: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command("index-status")
def index_status(
    index_path: Annotated[
        Path | None,
        typer.Option("--index-path", help="Persistent local Qdrant directory."),
    ] = None,
) -> None:
    """Verify an existing index and print its compatibility manifest."""
    settings = AppSettings()
    _echo_model(read_index_manifest(settings.index_config(path=index_path)))


def main() -> None:
    """Run the CLI with concise operational errors by default."""
    try:
        app()
    except Exception as error:
        if os.getenv("VEHICLE_SPECS_TRACEBACK") == "1":
            raise
        typer.echo(f"error: {type(error).__name__}: {error}", err=True)
        raise SystemExit(1) from None
