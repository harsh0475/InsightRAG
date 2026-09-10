"""Deterministic mock LLM provider for tests, local offline development, and CI/CD."""
import re
from typing import Optional
from backend.app.services.llm.base import BaseLLMProvider


class MockLLMProvider(BaseLLMProvider):
    """Deterministic LLM simulation for factual answering, citation formatting, and refusal."""

    def __init__(self, model_name: str = "mock-gpt-4o-mini"):
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Simulate grounded LLM generation based on context keywords and citation tagging."""
        prompt_lower = prompt.lower()

        # Check if context is completely empty or explicitly marked insufficient
        if "no relevant context" in prompt_lower or "context is empty" in prompt_lower:
            return "I do not have sufficient information in the provided context to answer this question."

        # Extract available chunk IDs from the prompt context block [ID: doc_xxx_c0000]
        chunk_ids = re.findall(r"\[ID:\s*([a-zA-Z0-9_\-]+)\]", prompt)
        if not chunk_ids:
            return "I do not have sufficient information in the provided context to answer this question."

        # Extract user question specifically to prevent context false-positives
        q_match = re.search(r"=== USER QUESTION ===\s*(.*?)\s*(?:Provide a grounded|\Z)", prompt, re.DOTALL)
        user_question = q_match.group(1).lower().strip() if q_match else prompt_lower

        # 1. OAuth queries
        if any(w in user_question for w in ["oauth", "token", "pkce", "grant"]):
            # Prefer OAuth chunks if available
            oauth_cids = [c for c in chunk_ids if "oauth" in c.lower()]
            cid = oauth_cids[0] if oauth_cids else chunk_ids[0]
            return (
                f"OAuth 2.0 uses PKCE for public clients. Access tokens have a 15-minute expiration, "
                f"while refresh tokens are valid for 30 days and enforce Refresh Token Rotation [{cid}]."
            )

        # 2. Kubernetes queries
        if any(w in user_question for w in ["kubernetes", "k8s", "control plane", "etcd", "pod", "cluster"]):
            k8s_cids = [c for c in chunk_ids if "k8s" in c.lower() or "guide" in c.lower()]
            cid = k8s_cids[0] if k8s_cids else chunk_ids[0]
            return (
                f"The Kubernetes control plane manages worker nodes and pods. Its primary components include "
                f"kube-apiserver, etcd (key-value backing store), kube-scheduler, and kube-controller-manager [{cid}]."
            )

        # 3. Redis / Caching queries
        if any(w in user_question for w in ["cache", "redis", "ttl", "stampede"]):
            arch_cids = [c for c in chunk_ids if "arch" in c.lower() or "redis" in c.lower()]
            cid = arch_cids[0] if arch_cids else chunk_ids[0]
            return (
                f"To prevent cache stampedes, distributed services use jittered expiration (5% to 15% random jitter) "
                f"and probabilistic early recomputation via the XFetch algorithm [{cid}]."
            )

        # 4. If query does not match facts in the corpus, refuse
        return "I do not have sufficient information in the provided context to answer this question."
