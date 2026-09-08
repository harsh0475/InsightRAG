# InsightRAG Architecture Specification

## 1. System Vision & Purpose

**InsightRAG** is an enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to provide verifiable, grounded answers over technical document corpora. The system prioritizes:
1. **Grounded Generation**: Zero tolerance for ungrounded claims or hallucinations; every factual statement maps back to retrieved source chunks.
2. **Retrieval Quality**: Hybrid search combining dense semantic embeddings with sparse lexical BM25 matching, fused via Reciprocal Rank Fusion (RRF) and refined by cross-encoder rerankers.
3. **Traceability & Transparency**: Full pipeline inspection (debug mode) revealing raw vector scores, BM25 scores, fusion ranks, rerank scores, and citation mappings.
4. **Empirical Evaluation**: Systematic measurement of retrieval (Recall@K, Hit Rate, MRR) and generation (Faithfulness, Answer Relevance).

---

## 2. High-Level Architecture

```
                                  +-----------------------------+
                                  |     Enterprise Corpora      |
                                  |   (PDF, Markdown, TXT)     |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |     Document Ingestion      |
                                  |   Parsing & Sanitization    |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |     Chunking & Metadata     |
                                  |  Fixed-size + overlap /     |
                                  |  Doc ID, Page, Section      |
                                  +--------------+--------------+
                                                 |
                        +------------------------+------------------------+
                        |                                                 |
                        v                                                 v
             +---------------------+                           +---------------------+
             | Dense Representations|                           | Inverted BM25 Index |
             | Embeddings (Provider)|                           | Term Frequencies    |
             +----------+----------+                           +----------+----------+
                        |                                                 |
                        v                                                 |
             +---------------------+                                      |
             | PostgreSQL + pgvector|                                     |
             | HNSW Index (Cosine) |                                      |
             +----------+----------+                                      |
                        |                                                 |
                        +------------------------+------------------------+
                                                 |
                                     (Query & Search Phase)
                                                 |
                                                 v
+------------------+              +-----------------------------+
|   User Query     | ---------->  |       Query Rewriter        |  (Contextualization for
+------------------+              |  (Conversation History Aware)|   follow-up questions)
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |      Hybrid Retrieval       |
                                  |  Dense Vector Search (Top-N)|
                                  |  + BM25 Keyword Search(Top-N|
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Reciprocal Rank Fusion(RRF) |
                                  |   Candidate Pool (Top 20)   |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |   Cross-Encoder Reranker    |
                                  |   Deep Relevance Scoring    |
                                  |      Top K Selection (5)    |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Context Builder & Prompting |
                                  | System instructions + chunks|
                                  | + strict citation contracts |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |    LLM Generator (OpenAI)   |
                                  |    Grounded Answer Output   |
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  | Grounded Answer + Citations |
                                  |       Evaluation Log        |
                                  +-----------------------------+
```

---

## 3. Core Component Breakdown

### 3.1 Document Ingestion & Chunking
- **Parsers**: Native PDF parsing (page boundary preservation), Markdown header-aware parsing, and structured TXT parsing.
- **Sanitization**: Whitespace normalization, encoding repair, header/footer noise removal.
- **Chunking Engine**: Sliding window chunker with configurable token/word size and overlap. Each chunk maintains rich provenance metadata (`document_id`, `chunk_id`, `page_number`, `section_title`, `token_count`).

### 3.2 Storage & Indexing Layer
- **Relational Storage**: PostgreSQL for document entities, ingestion runs, and conversation histories.
- **Vector Storage**: `pgvector` extension for storing high-dimensional embeddings.
  - **Index**: HNSW (Hierarchical Navigable Small World) with cosine distance (`vector_cosine_ops`) for sub-millisecond approximate nearest-neighbor queries.
- **BM25 Inverted Index**: Stored in-memory or persisted inverted index structure mapping vocabulary terms to chunk frequencies ($f(q_i, D)$), chunk lengths, and average collection length.

### 3.3 Retrieval Engine
- **Dense Retriever**: Generates query vector via `EmbeddingProvider` and computes vector similarity against indexed chunks.
- **Sparse Retriever**: Tokenizes query and computes BM25 relevance scores over the corpus.
- **Reciprocal Rank Fusion (RRF)**: Combines candidate lists independently of their raw score scales using rank positions:
  $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
  where $k=60$ is a smoothing constant and $r_m(d)$ is the 1-based rank of document $d$ in retriever $m$.
- **Cross-Encoder Reranker**: Performs full attention between query and candidate chunk pairs $(Q, C_i)$ to re-score the top 20 candidates down to top 5 high-precision chunks.

### 3.4 Generation & Citation Layer
- **Prompt Engineering**: System prompt enforcing strict grounding: "Answer solely from the provided context. If the context is insufficient, state that clearly. Cite every claim with [DocID:ChunkID]."
- **LLM Abstraction**: Generic `LLMProvider` interface compatible with OpenAI, Anthropic, or local vLLM/Ollama endpoints.

### 3.5 Evaluation & Observability
- **Retrieval Metrics**: Hit Rate@K, Recall@K, Mean Reciprocal Rank (MRR).
- **Generation Metrics**: Faithfulness (fraction of generated claims supported by context) and Answer Relevance.
- **System Metrics**: Latency decomposition (embedding, vector search, BM25, RRF, reranking, LLM time-to-first-token, total generation time).

---

## 4. Technology Stack & Tradeoffs

| Component | Technology | Rationale & Tradeoffs |
|---|---|---|
| **API Framework** | FastAPI (Python 3.12+) | Async native, Pydantic validation, automatic OpenAPI documentation, high developer velocity. |
| **Vector DB** | PostgreSQL + pgvector | Eliminates distributed system complexity. Keeps metadata, relational data, and vectors in a single ACID-compliant transactional store. |
| **Indexing** | HNSW (Cosine metric) | Fast retrieval with high recall; slightly higher memory footprint during index build than IVFFlat, but significantly better query latency under load. |
| **BM25 Engine** | Rank-BM25 / Custom Inverted Index | Transparent, inspectable TF-IDF/BM25 scoring without blackbox opacity. |
| **Reranking** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | High semantic precision by computing joint cross-attention over $(Query, Context)$ pairs. |
| **Frontend** | Next.js + Tailwind CSS | Fast SSR/SPA, clean component-driven UI with dedicated Retrieval Debug inspector. |

