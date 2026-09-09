"""CLI tool to index documents and perform semantic vector search with score inspection."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.document import DocumentChunk
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.retrieval_service import RetrievalService


def index_sample_documents(retrieval_service: RetrievalService, kb_dir: str = "knowledge_base"):
    """Scan knowledge_base directory and index all sample documents."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        print(f"Knowledge base directory '{kb_dir}' not found.")
        return 0

    ingestion_service = IngestionService(chunk_size=300, chunk_overlap=30)
    all_chunks: list[DocumentChunk] = []

    files = list(kb_path.glob("*.pdf")) + list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))
    print(f"\n[*] Scanning {len(files)} document(s) from '{kb_dir}'...")

    for fpath in files:
        with open(fpath, "rb") as f:
            content = f.read()
        try:
            result = ingestion_service.ingest_document(content, fpath.name, source_path=str(fpath))
            print(f"  + Ingested '{fpath.name}': {result.total_chunks} chunks ({result.total_tokens} tokens)")
            all_chunks.extend(result.chunks)
        except Exception as e:
            print(f"  - Failed to ingest '{fpath.name}': {e}", file=sys.stderr)

    if all_chunks:
        print(f"\n[*] Generating dense embeddings and indexing {len(all_chunks)} chunks...")
        indexed = retrieval_service.index_chunks(all_chunks)
        print(f"[OK] Successfully indexed {indexed} chunks into vector store.\n")
        return indexed
    return 0


def execute_search(retrieval_service: RetrievalService, query: str, top_k: int = 3):
    """Run vector search and display ranked results with similarity scores and provenance."""
    print("=" * 80)
    print(f" SEMANTIC VECTOR SEARCH: \"{query}\" (Top-K: {top_k})")
    print("=" * 80)

    results = retrieval_service.retrieve(query=query, top_k=top_k)

    if not results:
        print(" No matching chunks found.")
        return

    for i, res in enumerate(results, 1):
        meta = res.metadata
        print(f" [{i}] COSINE SIMILARITY SCORE: {res.score:.4f} | CHUNK: {res.chunk_id}")
        print(f"     Document : {res.document_name}")
        if meta.page_number:
            print(f"     Page     : {meta.page_number}")
        if meta.section:
            print(f"     Section  : {meta.section}")
        print(f"     Tokens   : {meta.token_count} | Position: {meta.chunk_position}")
        print(f"     Text Snippet:\n     \"{res.text[:280]}...\"")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="InsightRAG Vector Search & Retrieval Inspector"
    )
    parser.add_argument("--query", type=str, help="Search query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--kb-dir", type=str, default="knowledge_base", help="Knowledge base directory to index")
    parser.add_argument("--index-only", action="store_true", help="Index documents without querying")

    args = parser.parse_args()
    retrieval_service = RetrievalService()

    # Index sample documents
    index_sample_documents(retrieval_service, kb_dir=args.kb_dir)

    if args.index_only:
        return

    if args.query:
        execute_search(retrieval_service, args.query, top_k=args.top_k)
    else:
        # Interactive mode
        print("Entering interactive vector search mode. Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                user_query = input("Enter search query > ").strip()
                if not user_query:
                    continue
                if user_query.lower() in ("exit", "quit", "q"):
                    break
                execute_search(retrieval_service, user_query, top_k=args.top_k)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
