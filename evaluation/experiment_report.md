# InsightRAG: 4-Way Retrieval & Generation Evaluation Report

This report details the quantitative benchmarks comparing all 4 retrieval strategies across the gold-standard evaluation dataset.

## Summary Comparison Matrix

| Retrieval Strategy | Hit Rate@3 | Recall@3 | MRR | Faithfulness | Relevance | Refusal Acc | Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Vector Only** | 100.0% | 62.5% | 0.958 | 80.2% | 32.8% | 100.0% | 16.0ms |
| **2. BM25 Only** | 100.0% | 75.0% | 1.000 | 80.5% | 32.8% | 100.0% | 10.7ms |
| **3. Hybrid (RRF)** | 100.0% | 75.0% | 1.000 | 79.1% | 32.8% | 100.0% | 11.6ms |
| **4. Two-Stage Reranked** | 100.0% | 72.9% | 1.000 | 79.1% | 32.8% | 100.0% | 10.2ms |

## Metric Definitions
- **Hit Rate@K**: Proportion of queries where at least one chunk from the expected document was retrieved in Top-K.
- **Recall@K**: Proportion of crucial ground-truth keywords captured within the retrieved context.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank (1/rank) of the first relevant chunk.
- **Faithfulness (Groundedness)**: Proportion of claims/content words in the generated answer supported by retrieved context.
- **Answer Relevance**: Semantic concept overlap between the generated answer and ground-truth reference.
- **Refusal Accuracy**: Correct handling of out-of-domain unanswerable queries vs answerable queries.
