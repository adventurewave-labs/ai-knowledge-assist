from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def stream_client(mock_retriever):
    """A TestClient whose RAGChain.astream yields a fixed token sequence."""
    from app.ingestion.ingest import IngestPipeline
    from app.rag.chain import RAGChain

    mock_chain = MagicMock(spec=RAGChain)
    mock_chain.last_model_used = "gpt-4o-mini"

    async def fake_astream(question):
        for tok in ["Hello", ", ", "world"]:
            yield tok

    mock_chain.astream = fake_astream

    mock_pipeline = MagicMock(spec=IngestPipeline)

    with patch("app.main.VectorStoreRetriever", return_value=mock_retriever), patch(
        "app.main.RAGChain", return_value=mock_chain
    ), patch("app.main.IngestPipeline", return_value=mock_pipeline):
        from app.main import app

        with TestClient(app) as client:
            yield client


def test_stream_returns_event_stream(stream_client):
    resp = stream_client.get("/query/stream", params={"q": "hi"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_stream_sets_streaming_headers(stream_client):
    resp = stream_client.get("/query/stream", params={"q": "hi"})
    assert resp.headers.get("cache-control") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"


def test_stream_contains_data_tokens(stream_client):
    resp = stream_client.get("/query/stream", params={"q": "hi"})
    body = resp.text
    assert "data: Hello" in body
    assert "data: world" in body


def test_stream_final_event_is_done(stream_client):
    resp = stream_client.get("/query/stream", params={"q": "hi"})
    body = resp.text
    assert body.rstrip().endswith("data: [DONE]")


def test_stream_missing_q_returns_422(stream_client):
    resp = stream_client.get("/query/stream")
    assert resp.status_code == 422


def test_stream_rate_limit_applied(stream_client, monkeypatch):
    """A tiny limit causes the streaming endpoint to start returning 429."""
    from app.config import settings
    from app.rate_limit import limiter

    monkeypatch.setattr(settings, "RATE_LIMIT_RPM", 2)
    limiter.reset()

    headers = {"X-API-Key": "stream-ratelimit-key"}
    statuses = [
        stream_client.get(
            "/query/stream", params={"q": "hi"}, headers=headers
        ).status_code
        for _ in range(5)
    ]
    assert 429 in statuses, f"expected a 429 in {statuses}"
