import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded

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
from app.rate_limit import (
    limiter,
    rate_limit_exceeded_handler,
    rate_limit_value,
)

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
    version="0.2.0",
    lifespan=lifespan,
)

# Rate limiting (ADR-002). slowapi reads the limiter from app.state and uses
# the registered handler to render 429 responses.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# Serve the single-file web UI (ADR-004).
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
@limiter.limit(rate_limit_value)
async def query(request: Request, payload: QueryRequest) -> QueryResponse:
    chain: RAGChain = _state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    try:
        return chain.query(
            question=payload.question,
            top_k=payload.top_k,
            filters=payload.filters if payload.filters else None,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/query/stream", dependencies=[Depends(require_api_key)])
@limiter.limit(rate_limit_value)
async def query_stream(request: Request, q: str) -> StreamingResponse:
    """Stream an answer as Server-Sent Events (ADR-001).

    Emits ``data: <token>\\n\\n`` per chunk and a terminal ``data: [DONE]\\n\\n``.
    """
    chain: RAGChain = _state.get("chain")
    if chain is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    async def event_generator():
        try:
            async for token in chain.astream(q):
                yield f"data: {token}\n\n"
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "stream endpoint error",
                extra={"event": "stream_endpoint_error", "error": str(exc)},
            )
            yield f"data: [error] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    retriever: VectorStoreRetriever | None = _state.get("retriever")
    return HealthResponse(
        status="ok",
        version="0.2.0",
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
