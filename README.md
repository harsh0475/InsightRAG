# InsightRAG — Advanced Knowledge Intelligence System

> An enterprise-grade, portfolio-level Retrieval-Augmented Generation (RAG) system engineered for high-precision document intelligence, verifiable citations, hybrid retrieval, and rigorous evaluation.

---

## 1. Problem Statement

Standard Large Language Models (LLMs) suffer from knowledge cutoffs, domain gaps, and hallucinations when queried on private enterprise or proprietary technical documentation. While fine-tuning adjusts model style and vocabulary, it is computationally expensive, non-deterministic, and prone to catastrophic forgetting.

**InsightRAG** solves this through an end-to-end, grounded RAG architecture:
- Documents are parsed, chunked, and indexed with rich provenance metadata.
- Queries undergo hybrid retrieval (dense vector embeddings + sparse BM25 keyword matching).
- Candidate documents are merged via Reciprocal Rank Fusion (RRF) and re-scored via a Cross-Encoder reranker.
- Responses are generated with strict provenance citations (`[DocID:ChunkID]`) and verified against ungrounded hallucinations.

---

## 2. System Architecture

```
Technical Documents (PDF, MD, TXT)
               │
               ▼
      [Document Parsing]
               │
               ▼
     [Cleaning & Sanitizing]
               │
               ▼
 [Configurable Chunking + Overlap]
               │
               ▼
      [Provenance Metadata]
         /            \
        ▼              ▼
[Dense Embeddings]  [BM25 Inverted Index]
        │              │
        ▼              │
[pgvector / HNSW]      │
        │              │
        └───────┬──────┘
                │
         (Query Pipeline)
                │
                ▼
         [Query Rewriter]
                │
                ▼
      [Hybrid Search (Top-20)]
         Dense + BM25
                │
                ▼
 [Reciprocal Rank Fusion (RRF)]
                │
                ▼
  [Cross-Encoder Reranker (Top-5)]
                │
                ▼
     [Strict Context Builder]
                │
                ▼
     [OpenAI-Compatible LLM]
                │
                ▼
  [Grounded Answer + Citations]
                │
                ▼
  [Evaluation: Recall@K, MRR, Faithfulness]
```

---

## 3. Repository Structure

```
insightrag/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/              # API endpoints & routing (v1)
│   │   ├── core/             # Configuration, settings, logging
│   │   └── main.py           # Application entrypoint & lifespan
│   └── requirements.txt      # Python dependencies
├── frontend/                 # Next.js UI (Chat + Debug Inspector)
├── knowledge_base/           # Document repository (PDF, MD, TXT)
├── evaluation/               # Benchmark datasets & evaluation scripts
├── tests/                    # Unit and integration test suite
├── docs/                     # Architecture, roadmap, and interview guides
│   ├── architecture.md       # Comprehensive architectural deep-dive
│   └── roadmap.md            # 7-Day implementation plan
├── scripts/                  # Automation and migration scripts
├── docker/                   # Dockerfiles and docker-compose configurations
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions
└── README.md                 # Project documentation
```

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.12+ (or 3.13)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/InsightRAG.git
   cd InsightRAG
   ```

2. **Set up virtual environment**:
   ```powershell
   # Windows PowerShell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

5. **Run tests**:
   ```bash
   pytest
   ```

6. **Start the development server**:
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   - OpenAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Health Probe: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 5. Development Roadmap (7-Day Plan)

- **Day 1**: Milestone 0 (Architecture & Foundation) + Milestone 1 (Document Ingestion & Chunking)
- **Day 2**: Milestone 2 (Dense Embeddings & pgvector HNSW Search)
- **Day 3**: Milestone 3 (Baseline End-to-End RAG & Citation Generation)
- **Day 4**: Milestone 4 (Sparse BM25 & Hybrid Retrieval with RRF)
- **Day 5**: Milestone 5 (Cross-Encoder Reranking & Multi-turn Query Rewriting)
- **Day 6**: Milestone 6 (Empirical Evaluation: Hit Rate, Recall@K, MRR, Faithfulness)
- **Day 7**: Milestone 7 (Next.js UI with Retrieval Debug Inspector & Docker Deployment)

