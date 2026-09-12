"""CLI tool to query the InsightRAG pipeline with two-stage reranking and conversational query rewriting."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.chat import ChatMessage, ChatRole
from backend.app.schemas.document import DocumentChunk
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.rag.pipeline import BaselineRAGPipeline


def ensure_knowledge_base_indexed(pipeline: BaselineRAGPipeline, kb_dir: str = "knowledge_base"):
    """Index sample documents if vector store is empty."""
    if pipeline.retrieval_service.vector_store.count_chunks() > 0:
        return

    kb_path = Path(kb_dir)
    if not kb_path.exists():
        return

    ingestion_service = IngestionService(chunk_size=300, chunk_overlap=30)
    all_chunks: list[DocumentChunk] = []
    files = list(kb_path.glob("*.pdf")) + list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))

    print(f"[*] Indexing {len(files)} sample documents into vector store...")
    for fpath in files:
        with open(fpath, "rb") as f:
            content = f.read()
        try:
            res = ingestion_service.ingest_document(content, fpath.name, source_path=str(fpath))
            all_chunks.extend(res.chunks)
        except Exception:
            pass

    if all_chunks:
        pipeline.retrieval_service.index_chunks(all_chunks)
        print(f"[OK] Indexed {len(all_chunks)} chunks for InsightRAG.\n")


def print_rag_response(res):
    """Format and print RAG response."""
    print("=" * 80)
    print(f" ORIGINAL QUERY : {res.query}")
    if res.rewritten_query:
        print(f" REWRITTEN QUERY: {res.rewritten_query} (Contextualized from chat history)")
    print("=" * 80)
    print(
        f" MODEL   : {res.model_name} | TOTAL LATENCY: {res.execution_time_ms}ms"
        + (f" | RERANK: {res.reranker_latency_ms}ms" if res.reranker_latency_ms is not None else "")
    )
    print(f" STATUS  : {'Answered with Evidence' if res.has_sufficient_context else 'Insufficient Context Refusal'}")
    print(f" RERANKER: {'Applied (Top-20 candidates -> Top-K)' if res.reranker_applied else 'Disabled'}")
    print("-" * 80)
    print(" GROUNDED ANSWER:")
    print(res.answer)
    print("-" * 80)

    if res.citations:
        print(f" CITATIONS ({len(res.citations)}):")
        for i, cit in enumerate(res.citations, 1):
            page_str = f" | Page: {cit.page_number}" if cit.page_number else ""
            sec_str = f" | Section: {cit.section}" if cit.section else ""
            print(f"   [{i}] {cit.document_name}{page_str}{sec_str} [ID: {cit.chunk_id}]")
            print(f"       Snippet: \"{cit.snippet[:140]}...\"")
    else:
        print(" CITATIONS: None")

    print("-" * 80)
    if res.reranked_chunks:
        print(f" RERANKED CHUNKS ({len(res.reranked_chunks)}):")
        for rr in res.reranked_chunks:
            shift_symbol = f"+{rr.rank_delta}" if rr.rank_delta > 0 else str(rr.rank_delta)
            tag = "PROMOTED" if rr.rank_delta > 0 else ("DEMOTED" if rr.rank_delta < 0 else "KEPT")
            print(
                f"   [Rank {rr.final_rank} | {tag} ({shift_symbol})] "
                f"Score: {rr.rerank_score:.4f} (Prev: {rr.retrieval_score:.4f}, Rank {rr.initial_rank}) | "
                f"Doc: {rr.document_name} | Chunk: {rr.chunk_id}"
            )
    else:
        print(f" RETRIEVED CHUNKS ({len(res.retrieved_chunks)}):")
        for i, c in enumerate(res.retrieved_chunks, 1):
            print(f"   ({i}) Score: {c.score:.4f} | Chunk: {c.chunk_id} | Doc: {c.document_name}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="InsightRAG Advanced Q&A CLI (Two-Stage Reranking & Conversational Rewriting)"
    )
    parser.add_argument("--query", type=str, help="Question to ask the RAG pipeline")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to feed to LLM context")
    parser.add_argument("--candidate-pool", type=int, default=15, help="Number of candidate chunks for Stage 1")
    parser.add_argument("--disable-rerank", action="store_true", help="Disable Stage 2 cross-encoder reranking")
    args = parser.parse_args()

    pipeline = BaselineRAGPipeline()
    ensure_knowledge_base_indexed(pipeline)

    if args.query:
        res = pipeline.run(
            query=args.query,
            top_k=args.top_k,
            enable_reranking=not args.disable_rerank,
            candidate_pool_size=args.candidate_pool,
        )
        print_rag_response(res)
    else:
        print("InsightRAG Multi-Turn Conversational Q&A (Type 'exit' to quit)\n")
        chat_history: list[ChatMessage] = []

        while True:
            try:
                user_q = input("Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    break

                res = pipeline.run(
                    query=user_q,
                    top_k=args.top_k,
                    enable_reranking=not args.disable_rerank,
                    candidate_pool_size=args.candidate_pool,
                    chat_history=chat_history,
                )
                print_rag_response(res)

                # Record turn in history
                chat_history.append(ChatMessage(role=ChatRole.USER, content=user_q))
                chat_history.append(ChatMessage(role=ChatRole.ASSISTANT, content=res.answer))

            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
