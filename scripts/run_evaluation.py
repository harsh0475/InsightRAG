"""CLI evaluation runner producing quantitative benchmark reports and strategy comparisons."""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.schemas.document import DocumentChunk
from backend.app.services.evaluation.evaluator import EvaluationRunner
from backend.app.services.ingestion_service import IngestionService
from backend.app.services.rag.pipeline import BaselineRAGPipeline


def populate_knowledge_base(pipeline: BaselineRAGPipeline, kb_dir: str = "knowledge_base"):
    """Index sample documents if not already indexed."""
    kb_path = Path(kb_dir)
    if not kb_path.exists():
        print(f"[!] Knowledge base path {kb_dir} not found.")
        return

    if pipeline.retrieval_service.vector_store.count_chunks() > 0:
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
        print(f"[OK] Ingestion complete: {indexed} chunks indexed into vector and BM25 stores.\n")


def generate_markdown_report(results: dict, output_path: Path):
    """Generate structured Markdown experiment report."""
    lines = [
        "# InsightRAG: 4-Way Retrieval & Generation Evaluation Report",
        "",
        "This report details the quantitative benchmarks comparing all 4 retrieval strategies across the gold-standard evaluation dataset.",
        "",
        "## Summary Comparison Matrix",
        "",
        "| Retrieval Strategy | Hit Rate@3 | Recall@3 | MRR | Faithfulness | Relevance | Refusal Acc | Latency |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for name, r in results.items():
        lines.append(
            f"| **{name}** | {r.mean_hit_rate:.1%} | {r.mean_recall:.1%} | {r.mean_mrr:.3f} | "
            f"{r.mean_faithfulness:.1%} | {r.mean_relevance:.1%} | {r.refusal_accuracy:.1%} | {r.mean_latency_ms:.1f}ms |"
        )

    lines.extend([
        "",
        "## Metric Definitions",
        "- **Hit Rate@K**: Proportion of queries where at least one chunk from the expected document was retrieved in Top-K.",
        "- **Recall@K**: Proportion of crucial ground-truth keywords captured within the retrieved context.",
        "- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank (1/rank) of the first relevant chunk.",
        "- **Faithfulness (Groundedness)**: Proportion of claims/content words in the generated answer supported by retrieved context.",
        "- **Answer Relevance**: Semantic concept overlap between the generated answer and ground-truth reference.",
        "- **Refusal Accuracy**: Correct handling of out-of-domain unanswerable queries vs answerable queries.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Markdown report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="InsightRAG Quantitative Evaluation Runner")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K context chunks evaluated")
    parser.add_argument("--output-dir", default="evaluation", help="Directory to save experiment outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = EvaluationRunner()
    populate_knowledge_base(evaluator.pipeline)

    print("=" * 105)
    print(f" STARTING INSIGHTRAG 4-WAY COMPARATIVE EVALUATION BENCHMARK (Top-K={args.top_k})")
    print("=" * 105)

    start_bench = time.perf_counter()
    results = evaluator.compare_strategies(top_k=args.top_k)
    bench_duration = round(time.perf_counter() - start_bench, 2)

    print("\n" + "=" * 105)
    print(" EXPERIMENT BENCHMARK RESULTS")
    print("=" * 105)
    print(
        f"{'Strategy':<22} | {'Hit Rate':<10} | {'Recall':<8} | {'MRR':<8} | "
        f"{'Faithful':<10} | {'Relevance':<10} | {'Refusal':<9} | {'Latency':<9}"
    )
    print("-" * 105)

    json_export = {}
    for name, r in results.items():
        json_export[name] = {
            "retrieval_mode": r.retrieval_mode,
            "reranker_applied": r.reranker_applied,
            "mean_hit_rate": r.mean_hit_rate,
            "mean_recall": r.mean_recall,
            "mean_mrr": r.mean_mrr,
            "mean_faithfulness": r.mean_faithfulness,
            "mean_relevance": r.mean_relevance,
            "refusal_accuracy": r.refusal_accuracy,
            "mean_latency_ms": r.mean_latency_ms,
        }
        print(
            f"{name:<22} | {r.mean_hit_rate:>9.1%} | {r.mean_recall:>7.1%} | {r.mean_mrr:>7.3f} | "
            f"{r.mean_faithfulness:>9.1%} | {r.mean_relevance:>9.1%} | {r.refusal_accuracy:>8.1%} | {r.mean_latency_ms:>7.1f}ms"
        )

    print("=" * 105)
    print(f"Total benchmark runtime: {bench_duration}s across {len(evaluator.load_dataset())} test queries per strategy.\n")

    # Save JSON results
    json_path = out_dir / "experiment_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    print(f"[OK] JSON results saved to {json_path}")

    # Save Markdown report
    md_path = out_dir / "experiment_report.md"
    generate_markdown_report(results, md_path)


if __name__ == "__main__":
    main()

