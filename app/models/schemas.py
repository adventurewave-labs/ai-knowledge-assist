from typing import Optional

from pydantic import BaseModel, Field

_MAX_CONTENT_BYTES = 1_000_000  # 1 MB


class ErrorResponse(BaseModel):
    error_code: str
    message: str


class SourceDoc(BaseModel):
    content: str
    source: str
    score: Optional[float] = None


class IngestRequest(BaseModel):
    content: str = Field(..., max_length=_MAX_CONTENT_BYTES)
    source: str = Field(..., min_length=1, max_length=512)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    source: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, gt=0, le=100)
    filters: dict = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    model_used: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    vectorstore_ready: bool


class MetricsResponse(BaseModel):
    total_queries: int
    total_documents: int
    avg_latency_ms: float
    fallback_count: int
