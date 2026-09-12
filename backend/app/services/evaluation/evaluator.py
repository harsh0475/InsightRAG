"""Automated Evaluation Runner for quantitative & qualitative RAG benchmarking."""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from backend.app.schemas.evaluation import (
    EvalSample,
    EvaluationReport,
    SampleEvalResult,
)
from backend.app.schemas.retrieval import RetrievalMode
from backend.app.services.evaluation.metrics import (
    compute_generation_metrics,
    compute_retrieval_metrics,
)
from backend.app.services.rag.pipeline import BaselineRAGPipeline

logger = logging.getLogger("insightrag.evaluation.evaluator")


def _resolve_dataset_path() -> Path:
    # Check candidates from file location and working directory
    curr = Path(__file__).resolve()
    for parent in [curr.parent.parent.parent.parent.parent, curr.parent.parent.parent.parent, Path.cwd()]:
        candidate = parent / "evaluation" / "test_dataset.json"
        if candidate.exists():
            return candidate
    return curr.parent.parent.parent.parent.parent / "evaluation" / "test_dataset.json"


DEFAULT_DATASET_PATH = _resolve_dataset_path()


class EvaluationRunner:
    """Orchestrates benchmark runs across evaluation datasets and computes aggregate metrics."""

    def __init__(
        self,
        pipeline: Optional[BaselineRAGPipeline] = None,
        dataset_path: Optional[Path] = None,
    ):
        self.pipeline = pipeline or BaselineRAGPipeline()
        self.dataset_path = dataset_path or DEFAULT_DATASET_PATH
        self._latest_report: Optional[EvaluationReport] = None

    def load_dataset(self) -> List[EvalSample]:
        """Load benchmark queries and ground truth from JSON dataset."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [EvalSample(**item) for item in data]

    def evaluate_configuration(
        self,
        experiment_name: str = "Benchmark Evaluation",
        retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        enable_reranking: bool = True,
        top_k: int = 5,
        candidate_pool_size: int = 20,
    ) -> EvaluationReport:
        """Run full evaluation suite on the configured retrieval and reranking settings."""
        samples = self.load_dataset()
        sample_results: List[SampleEvalResult] = []

        total_hit_rate = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        total_faithfulness = 0.0
        total_relevance = 0.0
        total_refusal_correct = 0
        total_latency = 0.0

        for sample in samples:
            start_time = time.perf_counter()
            response = self.pipeline.run(
                query=sample.query,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                enable_reranking=enable_reranking,
                candidate_pool_size=candidate_pool_size,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Compute retrieval metrics
            retrieval_metrics = compute_retrieval_metrics(
                sample=sample,
                retrieved_chunks=response.retrieved_chunks,
                latency_ms=response.reranker_latency_ms or 0.0,
            )

            # Compute generation metrics
            is_refusal = not response.has_sufficient_context
            generation_metrics = compute_generation_metrics(
                sample=sample,
                answer=response.answer,
                retrieved_chunks=response.retrieved_chunks,
                citations=response.citations,
                is_refusal=is_refusal,
                latency_ms=response.execution_time_ms,
            )

            # Aggregate scores
            total_hit_rate += retrieval_metrics.hit_rate
            total_recall += retrieval_metrics.recall_at_k
            total_mrr += retrieval_metrics.reciprocal_rank
            total_faithfulness += generation_metrics.faithfulness
            total_relevance += generation_metrics.answer_relevance
            if generation_metrics.refusal_correctness:
                total_refusal_correct += 1
            total_latency += latency_ms

            retrieved_docs = list({c.document_name for c in response.retrieved_chunks})
            sample_results.append(
                SampleEvalResult(
                    query_id=sample.query_id,
                    query=sample.query,
                    expected_document=sample.expected_document,
                    is_answerable=sample.is_answerable,
                    retrieval_metrics=retrieval_metrics,
                    generation_metrics=generation_metrics,
                    retrieved_documents=retrieved_docs,
                    generated_answer=response.answer,
                    citations_count=len(response.citations),
                    is_refusal=is_refusal,
                )
            )

        n = len(samples)
        report = EvaluationReport(
            experiment_name=experiment_name,
            retrieval_mode=retrieval_mode.value,
            reranker_applied=enable_reranking,
            total_queries=n,
            mean_hit_rate=round(total_hit_rate / n, 4),
            mean_recall=round(total_recall / n, 4),
            mean_mrr=round(total_mrr / n, 4),
            mean_faithfulness=round(total_faithfulness / n, 4),
            mean_relevance=round(total_relevance / n, 4),
            refusal_accuracy=round(total_refusal_correct / n, 4),
            mean_latency_ms=round(total_latency / n, 2),
            sample_results=sample_results,
        )

        self._latest_report = report
        return report

    def compare_strategies(self, top_k: int = 3) -> Dict[str, EvaluationReport]:
        """Run evaluation across all 4 core retrieval strategies for empirical comparison."""
        configs = [
            ("1. Vector Only", RetrievalMode.VECTOR, False),
            ("2. BM25 Only", RetrievalMode.BM25, False),
            ("3. Hybrid (RRF)", RetrievalMode.HYBRID, False),
            ("4. Two-Stage Reranked", RetrievalMode.HYBRID, True),
        ]

        results: Dict[str, EvaluationReport] = {}
        for name, mode, rerank in configs:
            logger.info(f"Running strategy evaluation: {name}...")
            report = self.evaluate_configuration(
                experiment_name=name,
                retrieval_mode=mode,
                enable_reranking=rerank,
                top_k=top_k,
                candidate_pool_size=10,
            )
            results[name] = report

        return results

    @property
    def latest_report(self) -> Optional[EvaluationReport]:
        return self._latest_report

