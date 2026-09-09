# Vehicle Specification Extraction — Implementation Plan

## Goal

Build a clear, production-minded Python pipeline that extracts vehicle specifications from the supplied service manual using retrieval and an LLM. The submission should demonstrate PDF processing, chunking, embeddings, hybrid retrieval, structured extraction, provenance, and evaluation without adding an application API or background-job infrastructure.

## Scope decisions

- Build a synchronous CLI application and a small demonstration notebook.
- Use the supplied text-based service manual; OCR remains a documented future improvement.
- Keep manual-specific parsing and validation in plain Python.
- Use LangChain for document interchange, retrieval orchestration, prompting, and structured model output.
- Use Qdrant in local persistent mode so no vector database server is required.
- Use local dense and sparse embedding models by default.
- Keep the LLM provider configurable; support an entirely local model if required.
- Return evidence-backed JSON with explicit ambiguity and no-answer handling.

## Architecture

```text
PDF
  -> page-aware extraction and cleaning
  -> section/table-aware chunks with metadata
  -> persistent dense + sparse index
  -> hybrid retrieval
  -> optional cross-encoder reranking
  -> schema-constrained LLM extraction
  -> deterministic evidence validation
  -> JSON result
```

## Technology choices

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| Dependency/project management | `uv` and `pyproject.toml` |
| PDF extraction | PyMuPDF |
| Pipeline orchestration | LangChain Core |
| Dense embeddings | FastEmbed with BAAI/bge-small-en-v1.5 |
| Sparse embeddings | FastEmbed BM25 |
| Vector store | Qdrant local persistent mode |
| Reranking | Optional Sentence Transformers cross-encoder |
| Output validation | Pydantic |
| CLI | Typer |
| Tests | pytest |
| Static quality | Ruff and mypy |

## Proposed repository layout

```text
.
├── docs/
├── src/vehicle_specs/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── extractor.py
│   │   ├── cleaner.py
│   │   ├── tables.py
│   │   └── sections.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── chunker.py
│   │   └── io.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── documents.py
│   │   ├── embeddings.py
│   │   └── store.py
│   ├── retriever.py
│   ├── extractor.py
│   ├── validation.py
│   ├── pipeline.py
│   └── cli.py
├── notebooks/
│   └── demo.ipynb
├── eval/
│   └── gold_queries.jsonl
├── tests/
│   ├── unit/
│   └── integration/
├── artifacts/
├── .env.example
├── .gitignore
├── pyproject.toml
├── README.md
└── PLAN.md
```

## Implementation tracker

### 1. Project foundation

- [x] Create the package structure and `pyproject.toml`.
- [x] Lock dependencies with `uv`.
- [ ] Add typed configuration for file paths, model names, chunk settings, and retrieval limits.
- [ ] Add `.env.example` without secrets.
- [ ] Configure Ruff, mypy, and pytest.
- [ ] Create the Typer CLI skeleton.

### 2. PDF extraction and cleaning

- [x] Extract text blocks page-by-page with their coordinates.
- [x] Preserve PDF page numbers and discard validated recurring printed page labels.
- [x] Detect section identifiers, section names, article categories, and article headings.
- [x] Remove recurring headers, footers, dates, and `file:///` lines.
- [x] Normalize whitespace without destroying table row/column relationships.
- [x] Reconstruct high-confidence specification tables with raw-block fallback.
- [ ] Measure text coverage per page and flag suspiciously sparse pages.
- [ ] Persist cleaned page records as inspectable JSONL artifacts.
- [x] Add extraction and cleaning unit tests using representative pages.

### 3. Structure-aware chunking

- [x] Split first on manual articles and specification headings.
- [x] Keep torque, material, capacity, and part-number tables intact.
- [x] Apply a lexical-token fallback splitter only to oversized prose.
- [x] Carry section context into every chunk.
- [x] Add small overlap for prose chunks and avoid overlapping table rows.
- [x] Generate readable deterministic chunk IDs from source, article, page, and ordinal.
- [x] Store parser, cleaner, and chunker versions in chunk metadata.
- [x] Support atomic JSONL persistence for inspection and debugging.
- [x] Add tests proving that important specification tables are not split incorrectly.

### 4. Embedding and indexing

- [x] Create a LangChain-compatible local dense embedding adapter.
- [x] Configure FastEmbed BM25 sparse embeddings.
- [x] Create a persistent Qdrant collection with named dense and sparse vectors.
- [x] Store source, page, section, chunk ID, and pipeline versions as payload metadata.
- [x] Generate embeddings in batches.
- [x] Avoid inserting duplicate chunks within a single index build.
- [x] Add an explicit rebuild option for model or chunker changes.
- [x] Add a smoke test that closes and reopens the persistent index.

### 5. Retrieval and reranking

- [ ] Implement dense retrieval as a measurable baseline.
- [ ] Implement sparse retrieval as a measurable baseline.
- [ ] Implement hybrid retrieval using reciprocal-rank fusion.
- [ ] Retrieve a broad candidate set before narrowing the LLM context.
- [ ] Add optional cross-encoder reranking behind an interface.
- [ ] Support metadata filters such as section and front/rear applicability when reliable.
- [ ] Return retrieval scores and metadata in debug mode.
- [ ] Add tests for exact part numbers, paraphrased components, and ambiguous queries.

### 6. Structured LLM extraction

- [ ] Define Pydantic schemas for results, alternate units, applicability, and sources.
- [ ] Support `found`, `ambiguous`, and `not_found` statuses.
- [ ] Build a prompt that prohibits unsupported inference and calculations.
- [ ] Require all applicable values instead of selecting one arbitrary match.
- [ ] Require a page, section, and short evidence excerpt for every result.
- [ ] Use LangChain structured output rather than parsing free-form JSON manually.
- [ ] Record the model and prompt version with each response.
- [ ] Add deterministic post-generation validation:
  - [ ] The cited chunk was retrieved.
  - [ ] The evidence exists in the cited chunk.
  - [ ] The value and unit occur in the evidence.
  - [ ] Invalid or unsupported results are rejected.

### 7. Evaluation

- [ ] Create 20–30 manually verified gold queries.
- [ ] Include torque values, capacities, part numbers, dimensions, and no-answer cases.
- [ ] Include front/rear and component-type ambiguities.
- [ ] Include both exact manual terminology and natural-language paraphrases.
- [ ] Measure retrieval recall at 5 and 10.
- [ ] Measure component, value, and unit accuracy.
- [ ] Measure citation accuracy.
- [ ] Measure ambiguity and abstention accuracy.
- [ ] Produce an ablation comparison:

| Configuration | Recall@5 | Extraction accuracy | Citation accuracy |
|---|---:|---:|---:|
| Fixed chunks + dense retrieval | TBD | TBD | TBD |
| Section-aware chunks + dense retrieval | TBD | TBD | TBD |
| Section-aware chunks + hybrid retrieval | TBD | TBD | TBD |
| Hybrid retrieval + reranking | TBD | TBD | TBD |

- [ ] Save a machine-readable evaluation report and a readable Markdown summary.
- [ ] Add minimum regression thresholds to the test suite.

### 8. CLI and demonstration

- [ ] Implement the ingestion command.
- [ ] Implement the query command with JSON and human-readable output modes.
- [ ] Implement the evaluation command.
- [ ] Add a debug command or flag that prints retrieved chunks before extraction.
- [ ] Create a concise notebook that calls the package rather than duplicating pipeline logic.
- [ ] Demonstrate a successful, ambiguous, and no-answer query.

Planned commands:

```bash
uv run vehicle-specs ingest \
  --pdf "docs/sample-service-manual 1.pdf"

uv run vehicle-specs query \
  "Torque for front brake caliper guide pin bolts"

uv run vehicle-specs evaluate
```

### 9. Documentation and handoff

- [ ] Explain the architecture and data flow in `README.md`.
- [ ] Document installation and one-command reproduction steps.
- [ ] Explain chunking, retrieval, prompt, and model choices.
- [ ] Document measured results rather than only expected behavior.
- [ ] Document known limitations and failure modes.
- [ ] Add ideas for OCR, service deployment, access control, and scaling as future work.
- [ ] Ensure generated indexes, models, secrets, and large artifacts are ignored by Git.

## Output contract

```json
{
  "status": "found",
  "results": [
    {
      "component": "Brake caliper guide pin bolts",
      "spec_type": "torque",
      "value": "37",
      "unit": "Nm",
      "alternate_values": [
        {
          "value": "27",
          "unit": "lb-ft"
        }
      ],
      "applicability": "Front disc brake",
      "source": {
        "pdf_page": 636,
        "section": "206-03",
        "evidence": "Brake caliper guide pin bolts 37 27 —"
      }
    }
  ],
  "clarification": null,
  "pipeline_version": "1.0.0"
}
```

## Important test case from the supplied manual

The query `Torque for brake caliper bolts` must be treated as ambiguous because the manual contains multiple valid values, including:

- Front guide-pin bolts: 37 Nm on PDF page 636.
- Rear guide-pin bolts: 33 Nm on PDF page 652.
- Front anchor-plate bolts: 250 Nm on PDF page 636.
- Rear support-bracket bolts: 150 Nm on PDF page 652.

This case should verify retrieval recall, applicability handling, multiple-result extraction, and clarification behavior.

## Definition of done

- [ ] A clean environment can install and run the project using documented commands.
- [ ] The supplied PDF can be ingested synchronously and indexed persistently.
- [ ] Queries return validated structured data with page-level evidence.
- [ ] Ambiguous and unsupported queries are handled explicitly.
- [ ] The gold evaluation suite runs without manual steps.
- [ ] The README includes actual evaluation results and design trade-offs.
- [ ] Unit and integration tests pass.
- [ ] Ruff and mypy pass.

## Deferred scope

These remain valid production extensions but are intentionally excluded from the current assignment implementation:

- FastAPI or another HTTP API.
- Celery, Redis, or another background queue.
- PostgreSQL job and document metadata.
- S3/MinIO document storage.
- Authentication, authorization, and rate limiting.
- Distributed deployment and autoscaling.
- Streamlit or another UI.
- OCR and diagram/image understanding.
- Multi-tenant document isolation.
- Content hashing, duplicate-document detection, and checksum-based cache invalidation.
