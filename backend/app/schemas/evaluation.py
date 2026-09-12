"""Schemas for RAG quantitative and qualitative evaluation suite."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.retrieval import RetrievalMode


class EvalSample(BaseModel):
    """Single test case in gold-standard evaluation dataset."""
    query_id: str = Field(description="Unique query identifier (e.g. q001)")
    query: str = Field(description="User question to evaluate")
    expected_document: str = Field(description="Filename of expected source document")
    expected_keywords: List[str] = Field(
        default_factory=list, description="Crucial domain terms expected in retrieved chunks"
    )
    ground_truth_answer: str = Field(description="Reference human-verified answer")
    is_answerable: bool = Field(
        default=True, description="False if query is out-of-domain and should trigger refusal"
    )


class RetrievalEvalMetrics(BaseModel):
    """Retrieval quality metrics for a single query."""
    hit_rate: float = Field(description="1.0 if expected document found in Top-K, else 0.0")
    recall_at_k: float = Field(description="Fraction of expected keywords captured in retrieved context")
    reciprocal_rank: float = Field(description="1 / rank of first relevant chunk, 0.0 if not found")
    latency_ms: float = Field(description="Retrieval latency in milliseconds")


class GenerationEvalMetrics(BaseModel):
    """Generation quality metrics for a single query."""
    faithfulness: float = Field(
        description="Groundedness score: proportion of answer claims supported by retrieved context (0-1)"
    )
    answer_relevance: float = Field(
        description="Semantic relevance score of generated answer to the input query (0-1)"
    )
    citation_precision: float = Field(
        description="Proportion of citations that legitimately reference the expected source (0-1)"
    )
    refusal_correctness: bool = Field(
        description="True if system correctly refused an unanswerable query, or answered an answerable one"
    )
    latency_ms: float = Field(description="Generation latency in milliseconds")


class SampleEvalResult(BaseModel):
    """Full evaluation record for an individual test case."""
    query_id: str = Field(description="Sample query ID")
    query: str = Field(description="Input query")
    expected_document: str = Field(description="Expected source document")
    is_answerable: bool = Field(description="Whether query is answerable")
    retrieval_metrics: RetrievalEvalMetrics = Field(description="Retrieval stage metrics")
    generation_metrics: GenerationEvalMetrics = Field(description="Generation stage metrics")
    retrieved_documents: List[str] = Field(description="Documents retrieved in Top-K")
    generated_answer: str = Field(description="Generated answer from RAG pipeline")
    citations_count: int = Field(description="Number of citations produced")
    is_refusal: bool = Field(description="Whether the pipeline outputted a refusal")


class EvaluationReport(BaseModel):
    """Aggregated evaluation benchmark report across entire dataset."""
    experiment_name: str = Field(description="Identifier for this benchmark run")
    retrieval_mode: str = Field(description="Retrieval strategy evaluated: vector, bm25, or hybrid")
    reranker_applied: bool = Field(description="Whether cross-encoder reranking was enabled")
    total_queries: int = Field(description="Total number of evaluated queries")
    mean_hit_rate: float = Field(description="Average Hit Rate@K (0.0 to 1.0)")
    mean_recall: float = Field(description="Average Keyword Recall@K (0.0 to 1.0)")
    mean_mrr: float = Field(description="Mean Reciprocal Rank (MRR)")
    mean_faithfulness: float = Field(description="Average faithfulness/groundedness score (0.0 to 1.0)")
    mean_relevance: float = Field(description="Average answer relevance score (0.0 to 1.0)")
    refusal_accuracy: float = Field(description="Accuracy in handling unanswerable vs answerable queries")
    mean_latency_ms: float = Field(description="Mean end-to-end pipeline latency in milliseconds")
    sample_results: List[SampleEvalResult] = Field(default_factory=list, description="Per-sample evaluation results")


class EvaluationRunRequest(BaseModel):
    """Request payload to trigger an automated evaluation run."""
    experiment_name: str = Field(default="Evaluation Run", description="Name for the experiment run")
    retrieval_mode: RetrievalMode = Field(default=RetrievalMode.HYBRID, description="Retrieval strategy")
    enable_reranking: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
    top_k: int = Field(default=5, gt=0, le=20, description="Top-K context chunks passed to generation")
    candidate_pool_size: int = Field(default=20, gt=0, le=50, description="Stage 1 candidate pool size")

