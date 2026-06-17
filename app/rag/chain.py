import asyncio
import time
from typing import Any, AsyncIterator

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import settings
from app.llm.providers import get_fallback_llm, get_primary_llm
from app.logging_config import get_logger
from app.models.schemas import QueryResponse, SourceDoc
from app.rag.prompts import get_system_prompt
from app.rag.retriever import VectorStoreRetriever

logger = get_logger(__name__)

# Sentinel placed on the queue to signal that the producer thread has finished.
_STREAM_DONE = object()


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Bridges synchronous LangChain token callbacks into an asyncio.Queue.

    LangChain emits ``on_llm_new_token`` from the worker thread running the
    (sync) chain. We hand each token to the event loop's queue in a
    thread-safe manner so an async generator can consume them.
    """

    def __init__(self, queue: "asyncio.Queue", loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, token)


class RAGChain:
    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self._retriever = retriever
        self._total_queries = 0
        self._total_latency_ms = 0.0
        self._fallback_count = 0
        self._last_model_used = settings.LLM_MODEL
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
            model_used = settings.LLM_MODEL
        except Exception:
            try:
                model_used = settings.FALLBACK_LLM_MODEL
                self._fallback_count += 1
                llm = get_fallback_llm()
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({"context": context, "question": question})
            except Exception as fallback_err:
                logger.error(
                    "query failed: both primary and fallback LLMs errored",
                    extra={"event": "query_error", "question": question},
                )
                raise RuntimeError(
                    f"Both primary and fallback LLMs failed: {fallback_err}"
                ) from fallback_err

        self._last_model_used = model_used
        elapsed_ms = (time.monotonic() - start) * 1000
        self._total_queries += 1
        self._total_latency_ms += elapsed_ms

        logger.info(
            "query handled",
            extra={
                "event": "query",
                "question": question,
                "model_used": model_used,
                "num_sources": len(source_docs_raw),
                "latency_ms": round(elapsed_ms, 2),
            },
        )

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

    async def astream(self, question: str) -> AsyncIterator[str]:
        """Stream the answer token-by-token as an async generator.

        The underlying LangChain pipeline is synchronous, so it is run in a
        worker thread that pushes tokens onto an ``asyncio.Queue`` via
        :class:`StreamingCallbackHandler`. Falls back to the secondary LLM if
        the primary one fails before producing tokens.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        start = time.monotonic()

        lc_retriever = self._retriever.get_retriever()
        source_docs = lc_retriever.invoke(question)
        context = _format_docs(source_docs)

        system_prompt = get_system_prompt("default")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{question}"),
            ]
        )

        # Record the model before the first token is yielded so /metrics and
        # the UI reflect the active provider as soon as streaming begins.
        self._last_model_used = settings.LLM_MODEL

        def _produce() -> None:
            handler = StreamingCallbackHandler(queue, loop)
            try:
                llm = get_primary_llm(streaming=True)
                chain = prompt | llm | StrOutputParser()
                chain.invoke(
                    {"context": context, "question": question},
                    config={"callbacks": [handler]},
                )
            except Exception:
                try:
                    self._fallback_count += 1
                    self._last_model_used = settings.FALLBACK_LLM_MODEL
                    llm = get_fallback_llm(streaming=True)
                    chain = prompt | llm | StrOutputParser()
                    chain.invoke(
                        {"context": context, "question": question},
                        config={"callbacks": [handler]},
                    )
                except Exception as fallback_err:
                    logger.error(
                        "stream failed: both primary and fallback LLMs errored",
                        extra={"event": "stream_error", "question": question},
                    )
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        f"[error] {fallback_err}",
                    )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

        producer = loop.run_in_executor(None, _produce)

        try:
            while True:
                try:
                    token = await asyncio.wait_for(
                        queue.get(), timeout=settings.STREAM_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "stream timed out",
                        extra={"event": "stream_timeout", "question": question},
                    )
                    break
                if token is _STREAM_DONE:
                    break
                yield token
        finally:
            await producer

        elapsed_ms = (time.monotonic() - start) * 1000
        self._total_queries += 1
        self._total_latency_ms += elapsed_ms
        logger.info(
            "stream handled",
            extra={
                "event": "stream",
                "question": question,
                "model_used": self._last_model_used,
                "num_sources": len(source_docs),
                "latency_ms": round(elapsed_ms, 2),
            },
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

    @property
    def last_model_used(self) -> str:
        return self._last_model_used
