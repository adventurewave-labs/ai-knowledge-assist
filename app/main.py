import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import settings
from app.ingestion.ingest import IngestPipeline
from app.logging_config import configure_logging, request_id_var
from app.models.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)
from app.rag.chain import RAGChain
from app.rag.retriever import VectorStoreRetriever

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=[])

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

    logger.info("ARIA startup complete")
    yield

    retriever.close()
    _state.clear()
    logger.info("ARIA shutdown complete")


app = FastAPI(
    title="Project ARIA",
    description="RAG-powered knowledge assistant",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def _request_id_middleware(request: Request, call_next) -> Response:
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_var.set(req_id)
    start = time.monotonic()
    try:
        response: Response = await call_next(request)
    finally:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        request_id_var.reset(token)
        logger.info(
            "%s %s %s %.1fms",
            request.method,
            request.url.path,
            getattr(response, "status_code", 0),
            elapsed_ms,
        )
    response.headers["X-Request-ID"] = req_id
    return response


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error_code=code, message=message).model_dump(),
    )


@app.post("/v1/ingest", response_model=IngestResponse, tags=["ingestion"])
@limiter.limit("20/minute")
async def ingest_document(request: Request, body: IngestRequest) -> IngestResponse:
    pipeline: IngestPipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        return pipeline.ingest_document(
            content=body.content,
            source=body.source,
            metadata=body.metadata,
        )
    except Exception as exc:
        logger.error("Ingest failed for source %r: %s", body.source, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Ingestion failed") from exc


@app.post("/v1/query", response_model=QueryResponse, tags=["query"])
@limiter.limit("60/minute")
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    chain: RAGChain | None = _state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        return chain.query(
            question=body.question,
            top_k=body.top_k,
            filters=body.filters if body.filters else None,
        )
    except RuntimeError as exc:
        logger.error("Query failed (LLMs exhausted): %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="LLM service unavailable") from exc
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Query failed") from exc


@app.get("/v1/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    retriever: VectorStoreRetriever | None = _state.get("retriever")
    return HealthResponse(
        status="ok",
        version=app.version,
        vectorstore_ready=retriever is not None and retriever.is_ready,
    )


@app.get("/v1/metrics", response_model=MetricsResponse, tags=["ops"])
async def metrics() -> MetricsResponse:
    chain: RAGChain | None = _state.get("chain")
    retriever: VectorStoreRetriever | None = _state.get("retriever")

    return MetricsResponse(
        total_queries=chain.total_queries if chain else 0,
        total_documents=retriever.get_document_count() if retriever else 0,
        avg_latency_ms=chain.avg_latency_ms if chain else 0.0,
        fallback_count=chain.fallback_count if chain else 0,
    )
