import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_api_key
from app.config import settings
from app.ingestion.ingest import IngestPipeline
from app.logging_config import configure_logging, get_logger
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

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL)
    logger.info("starting Project ARIA", extra={"event": "startup"})

    retriever = VectorStoreRetriever()
    chain = RAGChain(retriever=retriever)
    chain.build_chain()
    pipeline = IngestPipeline(retriever=retriever)

    _state["retriever"] = retriever
    _state["chain"] = chain
    _state["pipeline"] = pipeline
    _state["start_time"] = time.monotonic()

    yield

    logger.info("shutting down Project ARIA", extra={"event": "shutdown"})
    _state.clear()


app = FastAPI(
    title="Project ARIA",
    description="RAG-powered knowledge assistant",
    version="0.1.1",
    lifespan=lifespan,
)

# B1: configurable CORS. Only enable credentialed access when origins are
# explicitly configured; otherwise fall back to a wildcard without credentials
# (browsers reject "*" together with credentials anyway).
_cors_origins = settings.cors_origins_list
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = round((time.monotonic() - start) * 1000, 2)
    logger.info(
        "request handled",
        extra={
            "event": "request",
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        },
    )
    return response


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
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


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
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
        version="0.1.1",
        vectorstore_ready=retriever is not None and retriever.is_ready,
    )


@app.get("/metrics", response_model=MetricsResponse, dependencies=[Depends(require_api_key)])
async def metrics() -> MetricsResponse:
    chain: RAGChain | None = _state.get("chain")
    retriever: VectorStoreRetriever | None = _state.get("retriever")

    total_queries = chain.total_queries if chain else 0
    avg_latency = chain.avg_latency_ms if chain else 0.0
    fallback_count = chain.fallback_count if chain else 0
    last_model_used = chain.last_model_used if chain else settings.LLM_MODEL
    total_documents = retriever.get_document_count() if retriever else 0

    return MetricsResponse(
        total_queries=total_queries,
        total_documents=total_documents,
        avg_latency_ms=avg_latency,
        fallback_count=fallback_count,
        last_model_used=last_model_used,
    )
