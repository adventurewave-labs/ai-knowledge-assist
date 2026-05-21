import logging
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.llm.providers import get_fallback_llm, get_primary_llm
from app.models.schemas import QueryResponse, SourceDoc
from app.rag.prompts import get_system_prompt
from app.rag.retriever import VectorStoreRetriever

logger = logging.getLogger(__name__)


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self._retriever = retriever
        self._total_queries = 0
        self._total_latency_ms = 0.0
        self._fallback_count = 0
        self._chain = None

    def build_chain(self, prompt_name: str = "default") -> None:
        system_prompt = get_system_prompt(prompt_name)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )
        lc_retriever = self._retriever.get_retriever()
        self._chain = (
            {
                "context": lc_retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | get_primary_llm()
            | StrOutputParser()
        )

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        start = time.monotonic()
        model_used = settings.LLM_MODEL
        answer = ""

        lc_retriever = self._retriever.get_retriever(top_k=top_k, filters=filters)
        source_docs_raw = lc_retriever.invoke(question)

        system_prompt = get_system_prompt("default")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )
        context = _format_docs(source_docs_raw)

        try:
            llm = get_primary_llm()
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": question})
        except Exception as primary_err:
            logger.warning("Primary LLM failed, trying fallback: %s", primary_err)
            try:
                model_used = settings.FALLBACK_LLM_MODEL
                self._fallback_count += 1
                llm = get_fallback_llm()
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
            except Exception as fallback_err:
                raise RuntimeError(
                    f"Both primary and fallback LLMs failed: {fallback_err}"
                ) from fallback_err

        elapsed_ms = (time.monotonic() - start) * 1000
        self._total_queries += 1
        self._total_latency_ms += elapsed_ms

        sources = [
            SourceDoc(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                score=doc.metadata.get("score", 0.0),
            )
            for doc in source_docs_raw
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            model_used=model_used,
            latency_ms=round(elapsed_ms, 2),
        )

    @property
    def total_queries(self) -> int:
        return self._total_queries

    @property
    def avg_latency_ms(self) -> float:
        if self._total_queries == 0:
            return 0.0
        return round(self._total_latency_ms / self._total_queries, 2)

    @property
    def fallback_count(self) -> int:
        return self._fallback_count
