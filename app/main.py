import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ingestion.ingest import IngestPipeline
from app.models.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from app.rag.chain import RAGChain
from app.rag.retriever import VectorStoreRetriever

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = VectorStoreRetriever()
    chain = RAGChain(retriever=retriever)
    chain.build_chain()
    pipeline = IngestPipeline(retriever=retriever)

    _state["retriever"] = retriever
    _state["chain"] = chain
    _state["pipeline"] = pipeline
    _state["start_time"] = time.monotonic()

    yield

    _state.clear()


app = FastAPI(
    title="Project ARIA",
    description="RAG-powered knowledge assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest) -> IngestResponse:
    pipeline: IngestPipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        return pipeline.ingest_document(
            content=request.content,
            source=request.source,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    chain: RAGChain = _state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        return chain.query(
            question=request.question,
            top_k=request.top_k,
            filters=request.filters if request.filters else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    retriever: VectorStoreRetriever | None = _state.get("retriever")
    return HealthResponse(
        status="ok",
        version="0.1.0",
        vectorstore_ready=retriever is not None and retriever.is_ready,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    chain: RAGChain | None = _state.get("chain")
    retriever: VectorStoreRetriever | None = _state.get("retriever")

    total_queries = chain.total_queries if chain else 0
    avg_latency = chain.avg_latency_ms if chain else 0.0
    fallback_count = chain.fallback_count if chain else 0
    total_documents = retriever.get_document_count() if retriever else 0

    return MetricsResponse(
        total_queries=total_queries,
        total_documents=total_documents,
        avg_latency_ms=avg_latency,
        fallback_count=fallback_count,
    )
