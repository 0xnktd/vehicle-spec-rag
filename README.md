# Vehicle Specification Extraction

A local RAG pipeline that extracts torque values, capacities, dimensions,
materials, and part numbers from an automotive service-manual PDF. It uses
PyMuPDF, FastEmbed, Qdrant, LangChain, and an OpenAI-compatible vLLM server to
return structured answers with exact page-level evidence.

The project includes a CLI, a small Streamlit UI, and a verified evaluation set.
It does not require an application API or background queue.

See [`PIPELINE_DESIGN.md`](PIPELINE_DESIGN.md) for the design and evaluation
method.

## Setup

### Prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Docker with the Compose plugin
- NVIDIA Container Toolkit for LLM inference
- The supplied manual at `docs/sample-service-manual 1.pdf`

PDF ingestion and retrieval run on CPU. The LLM-backed query flow and UI require
an NVIDIA GPU; the included vLLM configuration is tuned for an 8 GB RTX 4060.

### 1. Install the project

Run from the repository root:

```bash
mkdir -p artifacts
test -f .env || cp .env.example .env
uv sync --frozen --all-groups
```

If `.env` already exists, do not overwrite it. The default configuration works
with the supplied manual and stores generated files under `artifacts/`.

### 2. Extract and index the PDF

```bash
uv run vehicle-specs ingest
uv run vehicle-specs index-status
```

The first run downloads the local embedding models and creates cleaned-page,
quality-report, chunk, and Qdrant index artifacts. To intentionally replace an
existing index, run:

```bash
uv run vehicle-specs ingest --rebuild
```

### 3. Test retrieval without an LLM

```bash
uv run vehicle-specs retrieve \
  "Torque for front brake caliper guide pin bolts" \
  --context-k 3
```

Use `--rerank` to apply the local cross-encoder. Run
`uv run vehicle-specs retrieve --help` for filtering and retrieval-mode options.

### 4. Start vLLM and query the full pipeline

First verify that Docker can access the GPU:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

Start the OpenAI-compatible model server:

```bash
docker compose up -d vllm
docker compose logs -f vllm
```

The initial startup downloads the vLLM image and model. Wait until the service is
healthy, then stop following the logs with `Ctrl+C` and run:

```bash
uv run vehicle-specs query \
  "Torque for front brake caliper guide pin bolts" \
  --rerank \
  --output human
```

JSON is the default output. Add `--debug` to include retrieved chunks and ranking
scores.

### 5. Start the UI

```bash
docker compose up -d --build ui
```

Open [http://localhost:8501](http://localhost:8501). Stop all services with:

```bash
docker compose down
```

Model caches and the generated `artifacts/` directory are preserved.

### Optional: run the CLI in Docker

```bash
docker compose --profile cli build app
docker compose --profile cli run --rm app ingest
docker compose --profile cli run --rm app index-status
docker compose --profile cli run --rm app query \
  "Torque for front brake caliper guide pin bolts" \
  --rerank
```

Start `vllm` before the containerized `query` command. If the host user is not
UID/GID `1000:1000`, set `LOCAL_UID` and `LOCAL_GID` in `.env`.

### Run the checks

```bash
uv run pytest
uv run ruff format --check src tests streamlit_app.py
uv run ruff check src tests streamlit_app.py
uv run mypy src
docker compose config --quiet
```

The current suite contains 112 passing tests.

## Evaluation results

The supplied 852-page manual produced 637 structure-aware chunks, including 100
table chunks. The evaluation set contains 30 manually verified questions: 29
answerable cases and one unsupported case.

### Retrieval

| Configuration | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Dense only | 0.897 | 0.931 | 0.868 |
| Sparse BM25 only | 1.000 | 1.000 | 0.836 |
| Dense + sparse RRF | 0.966 | 0.966 | 0.848 |
| RRF + cross-encoder reranking | **1.000** | **1.000** | **0.931** |

### End-to-end extraction

The final run used vLLM `v0.29.0`,
`RedHatAI/Qwen3.5-4B-quantized.w4a16`, reranking, prompt `1.4.0`, and pipeline
`1.5.0`.

| Metric | Result |
|---|---:|
| Evaluated queries | 30 |
| Exact results | 34 / 34 |
| Status accuracy | 1.000 |
| Ambiguity accuracy | 1.000 |
| Abstention accuracy | 1.000 |
| Component accuracy | 1.000 |
| Value accuracy | 1.000 |
| Unit accuracy | 1.000 |
| Citation accuracy | 1.000 |
| Result precision | 1.000 |
| Exact-response accuracy | **1.000** |
| Operational failures | 0 |

Detailed reports are available in
[`eval/retrieval_results.md`](eval/retrieval_results.md),
[`eval/retrieval_results.json`](eval/retrieval_results.json),
[`eval/vllm_results.md`](eval/vllm_results.md), and
[`eval/vllm_results.json`](eval/vllm_results.json).

These results describe the checked-in evaluation set and should not be treated
as a general benchmark for every automotive manual.
