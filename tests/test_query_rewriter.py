"""Tests for Conversational Query Rewriter service."""
import pytest
from backend.app.schemas.chat import ChatMessage, ChatRole
from backend.app.services.llm.mock_provider import MockLLMProvider
from backend.app.services.rag.query_rewriter import QueryRewriter


def test_query_rewriter_no_history():
    """Rewriter should return original query untouched if history is empty."""
    rewriter = QueryRewriter(llm_provider=MockLLMProvider())
    query = "What is Redis replication?"
    res = rewriter.rewrite(query=query, chat_history=None)
    assert res == query

    res_empty = rewriter.rewrite(query=query, chat_history=[])
    assert res_empty == query


def test_query_rewriter_anaphora_resolution():
    """Rewriter should substitute ambiguous pronoun with subject from prior conversation."""
    rewriter = QueryRewriter(llm_provider=MockLLMProvider())
    history = [
        ChatMessage(role=ChatRole.USER, content="What is Redis replication?"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Redis replication copies data from a master to replica nodes."),
    ]
    follow_up = "How does it handle node failover?"
    rewritten = rewriter.rewrite(query=follow_up, chat_history=history)

    # Under mock heuristic, 'it' is replaced with 'Redis replication'
    assert "redis replication" in rewritten.lower()
    assert "failover" in rewritten.lower()


def test_query_rewriter_standalone_query_intact():
    """Rewriter should leave already self-contained follow-ups intact."""
    rewriter = QueryRewriter(llm_provider=MockLLMProvider())
    history = [
        ChatMessage(role=ChatRole.USER, content="What is Redis replication?"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Master-replica asynchronous replication."),
    ]
    standalone = "Explain OAuth 2.0 PKCE flow in detail."
    rewritten = rewriter.rewrite(query=standalone, chat_history=history)
    assert rewritten == standalone


def test_query_rewriter_history_formatting():
    """Check that conversation turns are formatted cleanly."""
    rewriter = QueryRewriter(llm_provider=MockLLMProvider(), max_history_turns=2)
    history = [
        ChatMessage(role=ChatRole.USER, content="Turn 1"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Answer 1"),
        ChatMessage(role=ChatRole.USER, content="Turn 2"),
        ChatMessage(role=ChatRole.ASSISTANT, content="Answer 2"),
    ]
    formatted = rewriter._format_history(history)
    assert "Turn 1" not in formatted  # Truncated to last 2 turns
    assert "User: Turn 2" in formatted
    assert "Assistant: Answer 2" in formatted

