import pytest


def test_health_returns_ok(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "vectorstore_ready" in body


def test_health_vectorstore_ready(test_client):
    resp = test_client.get("/health")
    assert resp.json()["vectorstore_ready"] is True


def test_metrics_returns_structure(test_client):
    resp = test_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "total_queries" in body
    assert "total_documents" in body
    assert "avg_latency_ms" in body
    assert "fallback_count" in body


def test_ingest_valid_document(test_client):
    payload = {
        "content": "# Test\n\nSome content.",
        "source": "api_test.md",
        "metadata": {"author": "test"},
    }
    resp = test_client.post("/ingest", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "api_test.md"
    assert "document_id" in body
    assert "chunks_created" in body


def test_ingest_missing_content_returns_422(test_client):
    resp = test_client.post("/ingest", json={"source": "no_content.md"})
    assert resp.status_code == 422


def test_ingest_missing_source_returns_422(test_client):
    resp = test_client.post("/ingest", json={"content": "Some text."})
    assert resp.status_code == 422


def test_query_valid_question(test_client):
    payload = {"question": "What is Python?", "top_k": 3}
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "model_used" in body
    assert "latency_ms" in body


def test_query_missing_question_returns_422(test_client):
    resp = test_client.post("/query", json={"top_k": 5})
    assert resp.status_code == 422


def test_query_with_filters(test_client):
    payload = {
        "question": "What is RAG?",
        "top_k": 5,
        "filters": {"source": "docs.md"},
    }
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 200


def test_query_default_top_k(test_client):
    payload = {"question": "Default top_k test"}
    resp = test_client.post("/query", json=payload)
    assert resp.status_code == 200
