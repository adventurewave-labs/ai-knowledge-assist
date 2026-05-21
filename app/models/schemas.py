from pydantic import BaseModel, Field


class SourceDoc(BaseModel):
    content: str
    source: str
    score: float


class IngestRequest(BaseModel):
    content: str
    source: str
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    source: str


class QueryRequest(BaseModel):
    question: str
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
