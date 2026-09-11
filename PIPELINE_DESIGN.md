# Pipeline Design and Evaluation

## Goal

The system answers questions about vehicle specifications in a service-manual
PDF. It returns the value, unit, and exact source instead of generating a free
text answer.

The current scope is text-based PDFs. Images, diagrams, and scanned pages are
not processed.

## Pipeline design

```text
PDF -> clean pages -> create chunks -> build index -> retrieve evidence
    -> extract structured fields -> validate evidence -> return result
```

### 1. Extract and clean the PDF

PyMuPDF reads each page as ordered text blocks with coordinates. The cleanup
step:

- removes repeated page headers, footers, dates, and local file paths;
- repairs broken words and visual line wrapping;
- keeps lists and procedure steps readable;
- converts reliable specification tables to Markdown; and
- records the PDF page and section for later citation.

Tables are only reconstructed when their headers are clear. Uncertain tables
remain as normal text to avoid joining a value to the wrong column.

### 2. Create chunks

Pages are grouped by manual article. Prose is split into bounded chunks with a
small overlap, while tables remain whole. Each chunk stores:

- its text and a stable ID;
- source filename and PDF pages;
- section, category, and article information; and
- parser and chunker versions.

Keeping this information with the text makes every retrieved result traceable
to the manual.

### 3. Build the index

Each chunk receives two local search representations:

- **Dense embedding:** finds similar meaning when the wording differs.
- **Sparse BM25 embedding:** finds exact terms, identifiers, and numbers.

Both are stored in a persistent local Qdrant index. The index also stores its
model and pipeline versions so incompatible indexes are not opened by mistake.

### 4. Retrieve evidence

For each question, the system can run dense search, sparse search, or both. The
default combines both rankings using reciprocal-rank fusion (RRF). An optional
cross-encoder then reranks the combined candidates, and the best five chunks are
sent to extraction.

Section and front/rear filters can narrow the search when the question provides
that information.

### 5. Extract structured fields

LangChain sends the question and retrieved chunks to Qwen through the local
OpenAI-compatible vLLM server. The model must return the Pydantic schema rather
than prose.

The response contains:

- `status`: `found`, `ambiguous`, or `not_found`;
- component, specification type, value, and unit;
- any printed alternate units or applicability; and
- chunk ID, PDF page, section ID, and exact evidence.

Deterministic table rules fill clearly matching rows that the model may omit and
remove duplicate results.

### 6. Validate before returning

The model does not get the final say. Every result is checked against the
retrieved chunks. The cited chunk must exist, the page and section must match,
the evidence must be copied from that chunk, and every returned value and unit
must occur in the evidence.

Unsupported output fails validation instead of being returned. If several valid
specifications match an unclear question, the result is `ambiguous` and includes
a clarification question.

## Why this design

- Dense and sparse search cover both paraphrases and exact workshop terms.
- Whole-table chunks preserve the relationship between headers, items, and
  values.
- The LLM performs structured extraction, while normal Python code handles PDF
  cleanup and evidence checks that should be deterministic.
- All models and storage run locally; no external API is required.
- The CLI and Streamlit UI call the same pipeline, so they return the same
  validated result.

## Evaluation process

### Dataset

The supplied 852-page manual produced 637 chunks, including 100 table chunks.
The gold set contains 30 manually checked questions:

- 29 answerable questions;
- 1 question whose answer is not in the manual; and
- 34 expected specification results in total.

The questions cover torque, capacity, dimension, part number, paraphrases,
front/rear differences, ambiguity, and refusal to invent an answer.

### Retrieval evaluation

The same gold questions were run with four retrieval configurations. Retrieval
was measured on the 29 answerable questions using:

- **Recall@5:** how much expected evidence appears in the first five chunks;
- **Recall@10:** how much appears in the first ten chunks; and
- **MRR:** how early the first correct chunk appears.

| Configuration | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Dense only | 0.897 | 0.931 | 0.868 |
| Sparse BM25 only | 1.000 | 1.000 | 0.836 |
| Dense + sparse RRF | 0.966 | 0.966 | 0.848 |
| RRF + cross-encoder reranking | **1.000** | **1.000** | **0.931** |

Sparse search found exact manual terms reliably. Dense search helped with
paraphrases. RRF alone lowered one correct result because several weaker chunks
appeared in both searches. The cross-encoder restored that result and gave the
best overall ordering, so it was selected for the final run.

### End-to-end evaluation

The final evaluation ran all 30 questions through retrieval, reranking, Qwen,
normalization, and evidence validation.

- Server: vLLM `v0.29.0`
- Model: `RedHatAI/Qwen3.5-4B-quantized.w4a16`
- GPU: NVIDIA RTX 4060 with 8 GB VRAM
- Prompt version: `1.4.0`
- Pipeline version: `1.5.0`

| Metric | Result |
|---|---:|
| Evaluated questions | 30 |
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

An exact result requires the returned component to contain the expected name and
the type, value, unit, and PDF page to match. An exact response requires the
status and complete result set to match, with no extra results. The evaluation
also records each runtime failure and fails the command if any case cannot be
completed.

### Reproduce the evaluation

```bash
uv run vehicle-specs evaluate --mode dense
uv run vehicle-specs evaluate --mode sparse
uv run vehicle-specs evaluate --mode hybrid
uv run vehicle-specs evaluate --mode hybrid --rerank
uv run vehicle-specs evaluate --mode hybrid --rerank --with-llm
```

The LLM run requires the vLLM container. Detailed checked-in results are in
[`eval/retrieval_results.md`](eval/retrieval_results.md) and
[`eval/vllm_results.md`](eval/vllm_results.md).

## Result limits and next steps

The perfect end-to-end score applies only to this finite gold set, not to every
service manual. The next useful checks would be more manuals, harder negative
questions, and OCR evaluation for scanned pages. A server-hosted vector store
would also be needed for concurrent production workers.
