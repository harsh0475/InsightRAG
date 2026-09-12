"""Evaluation metrics engine computing retrieval and generation metrics."""
import re
from typing import List, Set

from backend.app.schemas.evaluation import EvalSample, GenerationEvalMetrics, RetrievalEvalMetrics
from backend.app.schemas.rag import Citation
from backend.app.schemas.retrieval import VectorSearchResult


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())


def compute_retrieval_metrics(
    sample: EvalSample,
    retrieved_chunks: List[VectorSearchResult],
    latency_ms: float = 0.0,
) -> RetrievalEvalMetrics:
    """Compute Hit Rate@K, Keyword Recall@K, and Reciprocal Rank for retrieved chunks.
    
    Args:
        sample: Evaluation sample with ground truth metadata.
        retrieved_chunks: List of context chunks retrieved in top-K.
        latency_ms: Latency taken by retrieval.
        
    Returns:
        RetrievalEvalMetrics instance.
    """
    if not sample.is_answerable:
        # For unanswerable queries, retrieval shouldn't find an expected document
        return RetrievalEvalMetrics(
            hit_rate=1.0,
            recall_at_k=1.0,
            reciprocal_rank=1.0,
            latency_ms=round(latency_ms, 2),
        )

    expected_doc = sample.expected_document.lower()
    hit = 0.0
    reciprocal_rank = 0.0

    for rank, chunk in enumerate(retrieved_chunks, start=1):
        chunk_doc = chunk.document_name.lower()
        if expected_doc in chunk_doc or chunk_doc in expected_doc:
            hit = 1.0
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

    # Keyword Recall: fraction of expected keywords present in retrieved context
    if sample.expected_keywords:
        combined_text = " ".join([c.text.lower() for c in retrieved_chunks])
        matched_keywords = sum(
            1 for kw in sample.expected_keywords if kw.lower() in combined_text
        )
        recall = matched_keywords / len(sample.expected_keywords)
    else:
        recall = 1.0

    return RetrievalEvalMetrics(
        hit_rate=hit,
        recall_at_k=round(recall, 4),
        reciprocal_rank=round(reciprocal_rank, 4),
        latency_ms=round(latency_ms, 2),
    )


def compute_generation_metrics(
    sample: EvalSample,
    answer: str,
    retrieved_chunks: List[VectorSearchResult],
    citations: List[Citation],
    is_refusal: bool,
    latency_ms: float = 0.0,
) -> GenerationEvalMetrics:
    """Compute Faithfulness, Answer Relevance, and Citation Precision.
    
    Args:
        sample: Ground truth sample.
        answer: Generated answer text.
        retrieved_chunks: Context chunks used by generator.
        citations: Citations parsed from answer.
        is_refusal: Whether system refused.
        latency_ms: Latency of generation.
        
    Returns:
        GenerationEvalMetrics instance.
    """
    # 1. Refusal correctness
    if not sample.is_answerable:
        refusal_correct = is_refusal
        # For correct refusal on unanswerable query: max scores
        if refusal_correct:
            return GenerationEvalMetrics(
                faithfulness=1.0,
                answer_relevance=1.0,
                citation_precision=1.0,
                refusal_correctness=True,
                latency_ms=round(latency_ms, 2),
            )
        else:
            # Hallucinated an answer for an unanswerable query
            return GenerationEvalMetrics(
                faithfulness=0.0,
                answer_relevance=0.0,
                citation_precision=0.0,
                refusal_correctness=False,
                latency_ms=round(latency_ms, 2),
            )
    else:
        refusal_correct = not is_refusal

    if is_refusal and sample.is_answerable:
        # False refusal when answer should have been found
        return GenerationEvalMetrics(
            faithfulness=1.0,
            answer_relevance=0.0,
            citation_precision=0.0,
            refusal_correctness=False,
            latency_ms=round(latency_ms, 2),
        )

    # 2. Faithfulness / Groundedness:
    # Measures the proportion of content words in the answer supported by retrieved context
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do",
        "does", "did", "this", "that", "these", "those", "it", "they", "we", "you",
        "of", "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "can", "will", "just", "should", "now"
    }

    answer_words = [w for w in _tokenize_words(answer) if w not in stop_words and len(w) > 2]
    context_text = " ".join([c.text.lower() for c in retrieved_chunks])

    if answer_words:
        grounded_words = sum(1 for w in answer_words if w in context_text)
        faithfulness = grounded_words / len(answer_words)
    else:
        faithfulness = 1.0

    # 3. Answer Relevance:
    # Overlap between ground truth answer key concepts and generated answer
    gt_words = [w for w in _tokenize_words(sample.ground_truth_answer) if w not in stop_words and len(w) > 2]
    if gt_words:
        covered_gt = sum(1 for w in gt_words if w in answer.lower())
        relevance = covered_gt / len(gt_words)
    else:
        relevance = 1.0

    # 4. Citation Precision:
    # Proportion of citations whose chunk matches the expected document
    if citations:
        valid_citations = 0
        expected_doc = sample.expected_document.lower()
        for cit in citations:
            if expected_doc in cit.document_name.lower():
                valid_citations += 1
        citation_precision = valid_citations / len(citations)
    else:
        citation_precision = 1.0 if not sample.is_answerable else 0.0

    return GenerationEvalMetrics(
        faithfulness=round(min(1.0, max(0.0, faithfulness)), 4),
        answer_relevance=round(min(1.0, max(0.0, relevance)), 4),
        citation_precision=round(min(1.0, max(0.0, citation_precision)), 4),
        refusal_correctness=refusal_correct,
        latency_ms=round(latency_ms, 2),
    )

