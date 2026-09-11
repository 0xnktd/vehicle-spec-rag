# Retrieval Evaluation Results

Run on 2026-09-09 against the supplied 852-page manual, 637 structure-aware
chunks, and 30 manually verified gold questions. Retrieval metrics use the 29
answerable questions; the unsupported question is evaluated by the extraction
abstention metric when `--with-llm` is enabled.

| Configuration | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| Dense only | 0.897 | 0.931 | 0.868 |
| Sparse BM25 only | 1.000 | 1.000 | 0.836 |
| Dense + sparse RRF | 0.966 | 0.966 | 0.848 |
| RRF + cross-encoder reranking | **1.000** | **1.000** | **0.931** |

The pure-RRF miss is the front wheel-speed-sensor torque table. BM25 ranks it
first, but it falls below the top ten after RRF because several weaker chunks
occur in both branches. The cross-encoder restores it and improves overall
ranking without changing the candidate generator.

Model-specific extraction was completed on 2026-09-10 with
`RedHatAI/Qwen3.5-4B-quantized.w4a16` served by the pinned vLLM Compose service.
All 30 responses and 34 expected results matched exactly with no operational
failures. See [`vllm_results.md`](vllm_results.md) for the runtime and full
aggregate metrics. Reproduce it with:

```bash
uv run vehicle-specs evaluate --rerank --with-llm
```
