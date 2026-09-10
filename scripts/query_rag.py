"""CLI tool to query the Baseline RAG pipeline with interactive and one-shot modes."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.document import DocumentChunk
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.rag.pipeline import BaselineRAGPipeline
from backend.app.services.retrieval_service import RetrievalService


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
        print(f"[OK] Indexed {len(all_chunks)} chunks for Baseline RAG.\n")


def print_rag_response(res):
    """Format and print RAG response."""
    print("=" * 80)
    print(f" QUESTION: {res.query}")
    print("=" * 80)
    print(f" MODEL   : {res.model_name} | LATENCY: {res.execution_time_ms}ms")
    print(f" STATUS  : {'Answered with Evidence' if res.has_sufficient_context else 'Insufficient Context Refusal'}")
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
    print(f" RETRIEVED CANDIDATES ({len(res.retrieved_chunks)}):")
    for i, c in enumerate(res.retrieved_chunks, 1):
        print(f"   ({i}) Score: {c.score:.4f} | Chunk: {c.chunk_id} | Doc: {c.document_name}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="InsightRAG Baseline Q&A CLI"
    )
    parser.add_argument("--query", type=str, help="Question to ask the RAG pipeline")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve for context")
    args = parser.parse_args()

    pipeline = BaselineRAGPipeline()
    ensure_knowledge_base_indexed(pipeline)

    if args.query:
        res = pipeline.run(query=args.query, top_k=args.top_k)
        print_rag_response(res)
    else:
        print("InsightRAG Baseline Q&A (Type 'exit' to quit)\n")
        while True:
            try:
                user_q = input("Question > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    break
                res = pipeline.run(query=user_q, top_k=args.top_k)
                print_rag_response(res)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()

