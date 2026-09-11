"""Benchmark and comparison script evaluating Vector Only vs BM25 Only vs Hybrid Retrieval."""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.document import DocumentChunk
from backend.app.schemas.retrieval import RetrievalMode
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.retrieval.hybrid import HybridRetriever


def index_sample_documents(hybrid_retriever: HybridRetriever, kb_dir: str = "knowledge_base"):
    """Index all knowledge base documents into both dense and BM25 stores."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        print(f"Error: Directory '{kb_dir}' not found.", file=sys.stderr)
        return 0

    ingestion = IngestionService(chunk_size=300, chunk_overlap=30)
    all_chunks: list[DocumentChunk] = []
    files = list(kb_path.glob("*.pdf")) + list(kb_path.glob("*.md")) + list(kb_path.glob("*.txt"))

    print(f"[*] Indexing {len(files)} documents into Hybrid Store (Dense Vector + BM25 Inverted Index)...")
    for fpath in files:
        with open(fpath, "rb") as f:
            content = f.read()
        try:
            res = ingestion.ingest_document(content, fpath.name, source_path=str(fpath))
            all_chunks.extend(res.chunks)
        except Exception as e:
            print(f"  - Failed to ingest '{fpath.name}': {e}", file=sys.stderr)

    if all_chunks:
        hybrid_retriever.index_chunks(all_chunks)
        print(f"[OK] Indexed {len(all_chunks)} chunks for comparative evaluation.\n")
        return len(all_chunks)
    return 0


def run_comparison(hybrid_retriever: HybridRetriever, query: str, top_k: int = 3):
    """Run query through Vector, BM25, and Hybrid modes, and display comparative breakdown."""
    print("=" * 95)
    print(f" COMPARATIVE RETRIEVAL EVALUATION FOR QUERY:\n \"{query}\"")
    print("=" * 95)

    # 1. Vector Only
    vec_results = hybrid_retriever.retrieve(query=query, top_k=top_k, mode=RetrievalMode.VECTOR)

    # 2. BM25 Only
    bm25_results = hybrid_retriever.retrieve(query=query, top_k=top_k, mode=RetrievalMode.BM25)

    # 3. Hybrid (RRF)
    hybrid_results = hybrid_retriever.retrieve(query=query, top_k=top_k, mode=RetrievalMode.HYBRID)

    header = f"{'Rank':<5} | {'Mode':<10} | {'Score':<10} | {'Chunk ID':<25} | {'Document & Provenance':<35}"
    divider = "-" * 95

    print(header)
    print(divider)

    for i in range(top_k):
        # Vector
        v_res = vec_results[i] if i < len(vec_results) else None
        v_str = f"[{v_res.chunk_id}] {v_res.document_name}" if v_res else "N/A"
        v_score = f"{v_res.score:.4f}" if v_res else "-"
        print(f"{i+1:<5} | {'VECTOR':<10} | {v_score:<10} | {v_res.chunk_id if v_res else '-':<25} | {v_res.document_name if v_res else '-':<35}")

        # BM25
        b_res = bm25_results[i] if i < len(bm25_results) else None
        b_score = f"{b_res.score:.4f}" if b_res else "-"
        print(f"{i+1:<5} | {'BM25':<10} | {b_score:<10} | {b_res.chunk_id if b_res else '-':<25} | {b_res.document_name if b_res else '-':<35}")

        # Hybrid
        h_res = hybrid_results[i] if i < len(hybrid_results) else None
        h_score = f"{h_res.score:.6f}" if h_res else "-"
        h_ranks = f"(v_rk:{h_res.vector_rank}, b_rk:{h_res.bm25_rank})" if h_res else ""
        print(f"{i+1:<5} | {'HYBRID RRF':<10} | {h_score:<10} | {h_res.chunk_id if h_res else '-':<25} | {h_res.document_name if h_res else '-'} {h_ranks}")
        print(divider)

    print("\n Top Hybrid Match Snippet:")
    if hybrid_results:
        top = hybrid_results[0]
        meta = top.metadata
        print(f" Chunk ID: {top.chunk_id} | Document: {top.document_name} | Page: {meta.page_number or 'N/A'}")
        print(f" \"{top.text[:300]}...\"\n")


def main():
    parser = argparse.ArgumentParser(description="InsightRAG Retrieval Comparison Tool")
    parser.add_argument("--query", type=str, help="Specific query to compare")
    parser.add_argument("--top-k", type=int, default=2, help="Number of results per mode to compare")
    args = parser.parse_args()

    retriever = HybridRetriever()
    index_sample_documents(retriever)

    # Benchmark test cases representing 3 canonical search challenges:
    benchmark_queries = [
        # 1. Exact term / code / alphanumeric query (BM25 excels)
        "RS256 algorithm and 43 characters code_verifier",
        # 2. Conceptual semantic query without exact words (Vector excels)
        "How does the master control node store cluster state and schedule pods?",
        # 3. Hybrid technical query (combines exact keywords and conceptual description)
        "Redis distributed caching jittered expiration and XFetch stampede mitigation",
    ]

    if args.query:
        run_comparison(retriever, args.query, top_k=args.top_k)
    else:
        print("Running comparative benchmark across 3 canonical retrieval challenges:\n")
        for q in benchmark_queries:
            run_comparison(retriever, q, top_k=args.top_k)


if __name__ == "__main__":
    main()

