# Model Card — ARIA Embedding Model

## Model Details

| Field | Value |
|-------|-------|
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | MiniLM (6-layer transformer) |
| Embedding dimensions | 384 |
| Max input length | 256 tokens |
| Normalization | L2 (unit vectors) |

## Intended Use

Encode document chunks and user queries into a shared vector space for semantic similarity retrieval within the ARIA RAG pipeline.

## Training Data

The base `all-MiniLM-L6-v2` model was trained by the sentence-transformers team on over 1 billion sentence pairs from diverse English-language sources including Wikipedia, news articles, Reddit, and scientific papers.

Fine-tuning on domain-specific corpora is supported via `sentence-transformers` training utilities. See `eval/benchmark.json` for the evaluation harness used to measure retrieval quality after fine-tuning.

## Performance

| Benchmark | Score |
|-----------|-------|
| SBERT STS-B | 0.8255 |
| Inference speed (CPU) | ~14 000 sentences/sec |

## Limitations

- Max 256 input tokens; longer text is truncated.
- Optimised for English; multilingual performance is degraded.
- Semantic similarity may not capture domain jargon without fine-tuning.

## Ethical Considerations

The model inherits biases present in its training corpora. Outputs should not be used as the sole basis for high-stakes decisions without human review.
