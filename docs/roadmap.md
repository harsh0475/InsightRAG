# InsightRAG: 7-Day Implementation Roadmap

This roadmap documents the 7-day incremental development cycle for **InsightRAG**, organized into verifiable milestones.

---

## Roadmap Overview

| Day | Milestone | Objective | Key Deliverables |
|---|---|---|---|
| **Day 1** | **Milestone 0 & 1** | Project Foundation & Document Ingestion | Repository scaffold, FastAPI base, config, health checks, PDF/MD/TXT parsers, configurable chunking engine with provenance metadata. |
| **Day 2** | **Milestone 2** | Embeddings & Vector Search | `EmbeddingProvider` abstraction, PostgreSQL + `pgvector` integration, HNSW cosine index, vector retrieval CLI & API. |
| **Day 3** | **Milestone 3** | Baseline RAG Pipeline | Context builder, grounding prompts, `LLMProvider` abstraction, baseline Q&A endpoint, citation tracking. |
| **Day 4** | **Milestone 4** | BM25 & Hybrid Retrieval | BM25 sparse keyword retriever, Reciprocal Rank Fusion (RRF) algorithm, comparative retrieval evaluation harness. |
| **Day 5** | **Milestone 5** | Reranking & Query Rewriting | Cross-encoder reranking (`ms-marco-MiniLM`), conversational query contextualizer/rewriter, multi-turn RAG. |
| **Day 6** | **Milestone 6** | Comprehensive Evaluation Suite | 30-50 curated test cases, automated retrieval metrics (Recall@K, MRR), generation metrics (Faithfulness, Relevance), experiment comparison matrix. |
| **Day 7** | **Milestone 7** | UI, Debug Inspector, Docker & Wrap-up | Next.js frontend with Retrieval Debug Mode, Docker Compose multi-container setup, complete interview prep guide. |

---

## Milestone Details

### Day 1: Foundation (Milestone 0) & Document Ingestion (Milestone 1)
- **Milestone 0**:
  - Project directory hierarchy.
  - Environment variable schema with `pydantic-settings`.
  - Structured logging with JSON/colorized formatter.
  - FastAPI health & readiness probe endpoints (`/api/v1/health`).
  - Unit tests verifying configuration loading and API routing.
- **Milestone 1**:
  - Parsers for `.pdf`, `.md`, `.txt`.
  - Document cleaner & text normalizer.
  - Configurable sliding-window chunker with token/word count & overlap.
  - Provenance metadata schemas (`document_id`, `page_number`, `section`, `chunk_id`, `chunk_index`).

### Day 2: Embeddings & Vector Retrieval (Milestone 2)
- Provider interface `EmbeddingProvider` (OpenAI / local HuggingFace / FastEmbed).
- Database schema for documents and vector chunks in PostgreSQL + `pgvector`.
- Approximate nearest neighbor search via HNSW cosine indexing.
- Search CLI & endpoint testing Top-K semantic matching.

### Day 3: Baseline RAG Pipeline (Milestone 3)
- `LLMProvider` interface (OpenAI-compatible client).
- Deterministic context construction with chunk boundaries and citation identifiers.
- System prompt enforcing strict factuality and citation generation (`[DocID:ChunkID]`).
- Baseline end-to-end question answering pipeline with citation resolution.

### Day 4: BM25 & Hybrid Retrieval (Milestone 4)
- BM25 tokenizer and inverted index scoring.
- Reciprocal Rank Fusion (RRF) algorithm combining dense vector ranks and sparse BM25 ranks.
- Evaluation harness comparing Dense vs. BM25 vs. Hybrid retrieval.

### Day 5: Cross-Encoder Reranking & Query Rewriting (Milestone 5)
- Cross-encoder reranker pipeline: candidate pool (Top 20) -> deep joint cross-attention -> Top 5.
- Query rewriting module: turns conversational history into a self-contained retrieval query.
- Pipeline logging of original vs. rewritten queries.

### Day 6: Quantitative & Qualitative Evaluation (Milestone 6)
- Gold standard test suite (30-50 curated questions with ground truth contexts and facts).
- Retrieval metrics: Hit Rate, Recall@K, Mean Reciprocal Rank (MRR).
- Generation metrics: Faithfulness, Answer Relevance, Citation Correctness.
- Automated experiment comparison reporting (`docs/experiments.md`).

### Day 7: UI, Debug Inspector & Containerization (Milestone 7)
- Next.js web application with chat interface, document management, and citations.
- "Retrieval Debug Mode": visualizes rewritten queries, dense/sparse candidate rankings, RRF scores, and rerank adjustments.
- Docker Compose multi-service deployment (`backend`, `frontend`, `postgres` with `pgvector`).
- Comprehensive technical interview documentation (`docs/interview-preparation.md`).

