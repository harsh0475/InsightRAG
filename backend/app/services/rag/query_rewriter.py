"""Conversational Query Rewriting Service for contextualizing follow-up questions."""
import logging
import re
from typing import List, Optional

from backend.app.schemas.chat import ChatMessage
from backend.app.services.llm import BaseLLMProvider, get_llm_provider
from backend.app.services.llm.mock_provider import MockLLMProvider

logger = logging.getLogger("insightrag.rag.query_rewriter")

REWRITE_SYSTEM_PROMPT = """You are an expert query reformulation assistant for a retrieval-augmented generation (RAG) system.
Given a multi-turn conversation history and a follow-up user question, your task is to reformulate the follow-up question into a single, self-contained, standalone search query that can be executed directly against a vector and lexical search engine.

RULES:
1. Resolve all pronouns (it, they, them, this, that, its, etc.) and references to previous entities or topics.
2. Preserve all specific domain keywords, acronyms, and technical constraints.
3. If the user question is ALREADY completely standalone and does not rely on previous context, output the question EXACTLY as-is.
4. Do NOT answer the question. Only output the reformulated search query.
5. Do NOT add unnecessary filler, explanations, or conversational pleasantries. Output ONLY the rewritten query text.
"""

REWRITE_USER_TEMPLATE = """Conversation History:
{history}

Current Question:
{query}

Standalone Search Query:"""


class QueryRewriter:
    """Reformulates multi-turn follow-up queries into self-contained search queries."""

    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None, max_history_turns: int = 6):
        self.llm_provider = llm_provider or get_llm_provider()
        self.max_history_turns = max_history_turns

    def _format_history(self, chat_history: List[ChatMessage]) -> str:
        """Format recent chat messages into a readable conversation transcript."""
        # Take the most recent turns up to max_history_turns
        recent = chat_history[-self.max_history_turns :]
        lines = []
        for msg in recent:
            role_label = msg.role.value.capitalize()
            lines.append(f"{role_label}: {msg.content.strip()}")
        return "\n".join(lines)

    def _heuristic_mock_rewrite(self, query: str, chat_history: List[ChatMessage]) -> str:
        """Deterministic resolution of pronouns for test and mock environments."""
        q_lower = query.lower()
        pronouns = [r"\bit\b", r"\bits\b", r"\bthey\b", r"\bthem\b", r"\bthis\b", r"\bthat\b", r"\bthese\b"]
        has_pronoun = any(re.search(p, q_lower) for p in pronouns)

        if not has_pronoun:
            return query

        # Find the most recent user message topic from history
        last_user_query = ""
        for msg in reversed(chat_history):
            if msg.role.value == "user":
                last_user_query = msg.content
                break

        if not last_user_query:
            return query

        # Extract subject from last query (e.g. "What is Redis replication?" -> "Redis replication")
        clean_last = re.sub(r"^(what is|what are|how does|explain|tell me about)\s+", "", last_user_query, flags=re.IGNORECASE)
        clean_last = clean_last.rstrip("?.,! ")

        if not clean_last:
            return query

        # Replace first pronoun with the subject
        rewritten = query
        for p in pronouns:
            if re.search(p, rewritten, flags=re.IGNORECASE):
                rewritten = re.sub(p, clean_last, rewritten, count=1, flags=re.IGNORECASE)
                break

        return rewritten

    def rewrite(self, query: str, chat_history: Optional[List[ChatMessage]] = None) -> str:
        """Reformulate query into a standalone retrieval query if conversation history exists.
        
        Args:
            query: The user's latest follow-up question.
            chat_history: List of preceding ChatMessage objects.
            
        Returns:
            A self-contained standalone search query string.
        """
        if not chat_history:
            return query.strip()

        # If running under MockLLMProvider, use deterministic heuristic rewriter
        if isinstance(self.llm_provider, MockLLMProvider):
            return self._heuristic_mock_rewrite(query, chat_history)

        formatted_history = self._format_history(chat_history)
        if not formatted_history.strip():
            return query.strip()

        prompt = REWRITE_USER_TEMPLATE.format(history=formatted_history, query=query)

        try:
            raw_result = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=REWRITE_SYSTEM_PROMPT,
                max_tokens=100,
                temperature=0.0,
            )
            rewritten = raw_result.strip().strip('"\'')
            # If the LLM returned an empty string or refusal, fallback to original query
            if not rewritten or len(rewritten) < 3:
                return query.strip()

            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed ({str(e)}). Falling back to original query.")
            return query.strip()

