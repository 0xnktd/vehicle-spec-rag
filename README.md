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

### Start everything

Place the supplied PDF at `docs/sample-service-manual 1.pdf`. Optionally copy
`.env.example` to `.env` to override the defaults.

Verify Docker GPU access once:

```bash
docker run --rm --gpus all \
  nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

Then start the complete application from the repository root:

```bash
docker compose up --build --wait
```

This one command builds the application image, creates or reuses the PDF index,
starts vLLM, and waits until the UI is healthy. The first run takes longer because
it downloads the embedding and LLM models and indexes the manual.

Open [http://localhost:8501](http://localhost:8501) after the command returns.
Use the following commands for logs and shutdown:

```bash
docker compose logs -f
docker compose down
```

Generated files remain under `artifacts/`, while model weights remain in Docker
volumes. Both are reused on the next start.

### Local CLI workflow

Install the locked Python environment and build the index:

```bash
uv sync --frozen --all-groups
uv run vehicle-specs ingest
uv run vehicle-specs index-status
```

Retrieval does not require vLLM:

```bash
uv run vehicle-specs retrieve \
  "Torque for front brake caliper guide pin bolts" \
  --context-k 3
```

With the Compose application running, query the full pipeline from the host:

```bash
uv run vehicle-specs query \
  "Torque for front brake caliper guide pin bolts" \
  --rerank \
  --output human
```

Use `docker compose --profile cli run --rm app <command>` to run the same CLI in
Docker. If the host user is not UID/GID `1000:1000`, set `LOCAL_UID` and
`LOCAL_GID` in `.env`.

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
