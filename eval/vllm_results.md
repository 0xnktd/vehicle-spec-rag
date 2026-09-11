# GPU-backed End-to-End Evaluation

Run on 2026-09-10 with the supplied 852-page service manual and 637 indexed
chunks.

## Runtime

- Command: `vehicle-specs evaluate --rerank --with-llm`
- GPU profile: NVIDIA RTX 4060, 8 GB
- Server: `vllm/vllm-openai:v0.29.0`
- Model: `RedHatAI/Qwen3.5-4B-quantized.w4a16`
- Model context: 8,192 tokens; one concurrent sequence; text-only mode
- Prompt version: `1.4.0`
- Pipeline version: `1.5.0`

## Retrieval

| Metric | Value |
|---|---:|
| Evaluated answerable queries | 29 |
| Recall@5 | 1.000 |
| Recall@10 | 1.000 |
| Mean reciprocal rank | 0.931 |

## Extraction

| Metric | Value |
|---|---:|
| Evaluated queries | 30 |
| Expected / returned results | 34 / 34 |
| Status accuracy | 1.000 |
| Ambiguity accuracy | 1.000 |
| Abstention accuracy | 1.000 |
| Component accuracy | 1.000 |
| Value accuracy | 1.000 |
| Unit accuracy | 1.000 |
| Citation accuracy | 1.000 |
| Result precision | 1.000 |
| Exact-result accuracy | 1.000 |
| Exact-response accuracy | 1.000 |
| Operational failures | 0 |

This is a regression result for the checked-in finite gold set, not a claim of
perfect accuracy on unseen manuals. The full per-case report is generated at
`artifacts/evaluation.json` and intentionally ignored because it contains every
retrieved response and can be reproduced from the command above.
