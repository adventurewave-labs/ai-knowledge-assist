from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.rag.chain import RAGChain


@pytest.fixture
def chain(mock_retriever):
    with patch("app.rag.chain.get_system_prompt", return_value="Answer: {context}"):
        c = RAGChain(retriever=mock_retriever)
        c.build_chain()
        return c


def _make_llm_mock(answer: str):
    llm = MagicMock()
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = answer
    llm.__or__ = MagicMock(return_value=chain_mock)
    return llm


def test_query_returns_response(mock_retriever):
    with patch("app.rag.chain.get_system_prompt", return_value="ctx: {context}"), patch(
        "app.rag.chain.get_primary_llm"
    ) as mock_primary:
        llm = MagicMock()
        runnable = MagicMock()
        runnable.invoke.return_value = "Test answer"
        mock_primary.return_value = llm

        with patch("app.rag.chain.StrOutputParser") as mock_parser_cls:
            mock_parser = MagicMock()
            mock_parser_cls.return_value = mock_parser

            with patch("app.rag.chain.ChatPromptTemplate") as mock_pt:
                prompt_instance = MagicMock()
                mock_pt.from_messages.return_value = prompt_instance
                composed = MagicMock()
                composed.invoke.return_value = "Mocked answer"
                prompt_instance.__or__ = MagicMock(return_value=composed)
                composed.__or__ = MagicMock(return_value=composed)

                c = RAGChain(retriever=mock_retriever)
                c.build_chain()
                resp = c.query(question="What is Python?")

        assert resp is not None


def test_metrics_increment(mock_retriever):
    with patch("app.rag.chain.get_system_prompt", return_value="ctx: {context}"), patch(
        "app.rag.chain.ChatPromptTemplate"
    ) as mock_pt, patch("app.rag.chain.get_primary_llm") as mock_primary, patch(
        "app.rag.chain.StrOutputParser"
    ):
        prompt_instance = MagicMock()
        mock_pt.from_messages.return_value = prompt_instance
        composed = MagicMock()
        composed.invoke.return_value = "Answer"
        prompt_instance.__or__ = MagicMock(return_value=composed)
        composed.__or__ = MagicMock(return_value=composed)
        mock_primary.return_value = MagicMock()

        c = RAGChain(retriever=mock_retriever)
        c.build_chain()

        assert c.total_queries == 0
        c.query(question="Q1")
        assert c.total_queries == 1
        c.query(question="Q2")
        assert c.total_queries == 2


def test_fallback_llm_used_on_primary_failure(mock_retriever):
    with patch("app.rag.chain.get_system_prompt", return_value="ctx: {context}"), patch(
        "app.rag.chain.ChatPromptTemplate"
    ) as mock_pt, patch(
        "app.rag.chain.get_primary_llm"
    ) as mock_primary, patch(
        "app.rag.chain.get_fallback_llm"
    ) as mock_fallback, patch(
        "app.rag.chain.StrOutputParser"
    ):
        prompt_instance = MagicMock()
        mock_pt.from_messages.return_value = prompt_instance

        fail_chain = MagicMock()
        fail_chain.invoke.side_effect = Exception("OpenAI error")
        success_chain = MagicMock()
        success_chain.invoke.return_value = "Fallback answer"

        or_calls = [0]

        def side_effect_or(other):
            or_calls[0] += 1
            if or_calls[0] <= 1:
                return fail_chain
            return success_chain

        prompt_instance.__or__ = MagicMock(side_effect=side_effect_or)
        fail_chain.__or__ = MagicMock(return_value=fail_chain)
        success_chain.__or__ = MagicMock(return_value=success_chain)

        mock_primary.return_value = MagicMock()
        mock_fallback.return_value = MagicMock()

        c = RAGChain(retriever=mock_retriever)
        c.build_chain()
        resp = c.query(question="Trigger fallback")

        assert c.fallback_count == 1
        assert resp.answer == "Fallback answer"


def test_avg_latency_zero_when_no_queries(mock_retriever):
    with patch("app.rag.chain.get_system_prompt", return_value="ctx: {context}"):
        c = RAGChain(retriever=mock_retriever)
        assert c.avg_latency_ms == 0.0


def test_fallback_count_starts_zero(mock_retriever):
    with patch("app.rag.chain.get_system_prompt", return_value="ctx: {context}"):
        c = RAGChain(retriever=mock_retriever)
        assert c.fallback_count == 0
