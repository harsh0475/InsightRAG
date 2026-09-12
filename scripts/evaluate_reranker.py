"""Benchmark script evaluating two-stage reranking and conversational query rewriting."""
import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.chat import ChatMessage, ChatRole
from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import RetrievalMode
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.rag.pipeline import BaselineRAGPipeline
from backend.app.services.rag.query_rewriter import QueryRewriter
from backend.app.services.reranker import get_reranker_provider


def populate_knowledge_base(pipeline: BaselineRAGPipeline, kb_dir: str = "knowledge_base"):
    """Index sample documents if not already indexed."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        print(f"[!] Knowledge base path {kb_dir} not found.")
        return

    ingestion_service = IngestionService(chunk_size=300, chunk_overlap=30)
    all_chunks: list[DocumentChunk] = []
    files = list(kb_path.glob("*.pdf")) + list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))

    print(f"[*] Ingesting and indexing {len(files)} documents for evaluation...")
    for fpath in files:
        with open(fpath, "rb") as f:
            content = f.read()
        try:
            res = ingestion_service.ingest_document(content, fpath.name, source_path=str(fpath))
            all_chunks.extend(res.chunks)
        except Exception as e:
            print(f"    Warning: Failed to parse {fpath.name}: {e}")

    if all_chunks:
        indexed = pipeline.retrieval_service.index_chunks(all_chunks)
        print(f"[OK] Ingestion complete: {indexed} chunks indexed across vector and BM25 stores.\n")


def run_reranker_benchmark(pipeline: BaselineRAGPipeline):
    """Run benchmark queries comparing Stage-1 Retrieval vs Stage-2 Cross-Encoder Reranking."""
    test_queries = [
        "What is the recommended Redis cache eviction policy for least frequently used items?",
        "How does OAuth 2.0 PKCE protect against authorization code injection?",
        "Explain Kubernetes pod disruption budgets and replica set scaling.",
        "What is the difference between Write-Through and Cache-Aside patterns?",
        "How are refresh tokens rotated upon exchange in OAuth?",
    ]

    print("=" * 95)
    print(" 1. TWO-STAGE RETRIEVAL BENCHMARK: STAGE 1 (HYBRID POOL) vs STAGE 2 (RERANKED TOP-5)")
    print("=" * 95)
    print(f"{'Query':<50} | {'Candidates':<10} | {'Rerank ms':<10} | {'Top-1 Shift':<12} | {'Promotions':<10}")
    print("-" * 95)

    reranker = get_reranker_provider()

    for query in test_queries:
        # 1. Retrieve candidate pool of 15
        s1_start = time.perf_counter()
        candidates = pipeline.hybrid_retriever.retrieve(query=query, top_k=15, mode=RetrievalMode.HYBRID)
        s1_time = round((time.perf_counter() - s1_start) * 1000, 2)

        from backend.app.schemas.retrieval import VectorSearchResult
        cand_objs = [
            VectorSearchResult(
                chunk_id=c.chunk_id,
                text=c.text,
                document_id=c.document_id,
                document_name=c.document_name,
                metadata=c.metadata,
                score=c.score,
            )
            for c in candidates
        ]

        # 2. Stage 2 Rerank to top 5
        s2_start = time.perf_counter()
        reranked = reranker.rerank(query=query, candidates=cand_objs, top_n=5)
        s2_time = round((time.perf_counter() - s2_start) * 1000, 2)

        # Check top-1 shift
        top1_shift = "No Change"
        if reranked:
            if reranked[0].initial_rank != 1:
                top1_shift = f"Rank {reranked[0].initial_rank} -> 1 (+{reranked[0].initial_rank - 1})"
            else:
                top1_shift = "Kept Rank 1"

        promotions = sum(1 for r in reranked if r.rank_delta > 0)
        short_q = query[:47] + "..." if len(query) > 50 else query
        print(f"{short_q:<50} | {len(candidates):<10} | {s2_time:>7.2f}ms | {top1_shift:<12} | {promotions:<10}")

    print("=" * 95 + "\n")


def run_conversational_rewrite_benchmark(pipeline: BaselineRAGPipeline):
    """Evaluate Conversational Query Rewriting on multi-turn ambiguous queries."""
    print("=" * 95)
    print(" 2. CONVERSATIONAL QUERY REWRITING BENCHMARK (ANAPHORA RESOLUTION)")
    print("=" * 95)

    scenarios = [
        {
            "history": [
                ChatMessage(role=ChatRole.USER, content="Explain OAuth 2.0 PKCE workflow."),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content="PKCE (Proof Key for Code Exchange) uses a code verifier and code challenge to secure mobile apps.",
                ),
            ],
            "follow_up": "How does it handle refresh token rotation?",
            "expected_entities": ["OAuth", "token", "rotation"],
        },
        {
            "history": [
                ChatMessage(role=ChatRole.USER, content="What is Redis Cluster architecture?"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content="Redis Cluster shards data across 16384 hash slots with master and replica nodes.",
                ),
            ],
            "follow_up": "What happens when a master node fails?",
            "expected_entities": ["Redis", "master", "fail"],
        },
        {
            "history": [
                ChatMessage(role=ChatRole.USER, content="How do Kubernetes Deployments manage rolling updates?"),
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content="Deployments use maxSurge and maxUnavailable to incrementally update pods.",
                ),
            ],
            "follow_up": "Can we roll them back if errors occur?",
            "expected_entities": ["Kubernetes", "roll", "back"],
        },
    ]

    rewriter = QueryRewriter(llm_provider=pipeline.llm_provider)

    for i, scen in enumerate(scenarios, 1):
        original = scen["follow_up"]
        rewritten = rewriter.rewrite(query=original, chat_history=scen["history"])

        print(f"[Scenario {i}]")
        print(f"   Original Follow-Up : {original}")
        print(f"   Rewritten Query    : {rewritten}")

        # Check retrieval difference
        res_orig = pipeline.hybrid_retriever.retrieve(query=original, top_k=3, mode=RetrievalMode.HYBRID)
        res_rewr = pipeline.hybrid_retriever.retrieve(query=rewritten, top_k=3, mode=RetrievalMode.HYBRID)

        orig_top_doc = res_orig[0].document_name if res_orig else "None"
        rewr_top_doc = res_rewr[0].document_name if res_rewr else "None"
        orig_score = f"{res_orig[0].score:.4f}" if res_orig else "N/A"
        rewr_score = f"{res_rewr[0].score:.4f}" if res_rewr else "N/A"

        print(f"   Without Rewrite Top Match: {orig_top_doc} (Score: {orig_score})")
        print(f"   With Rewrite Top Match   : {rewr_top_doc} (Score: {rewr_score})")
        print("-" * 95)

    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Two-Stage Reranker and Query Rewriter")
    parser.add_argument("--kb-dir", default="knowledge_base", help="Path to sample documents")
    args = parser.parse_args()

    pipeline = BaselineRAGPipeline()
    populate_knowledge_base(pipeline, kb_dir=args.kb_dir)

    run_reranker_benchmark(pipeline)
    run_conversational_rewrite_benchmark(pipeline)


if __name__ == "__main__":
    main()

