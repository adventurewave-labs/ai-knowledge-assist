"""Tests for the X-API-Key authentication on protected endpoints."""

import pytest

from app.config import settings


@pytest.fixture
def api_key(monkeypatch):
    key = "test-secret-key"
    monkeypatch.setattr(settings, "API_KEY", key)
    return key


def test_query_requires_key_when_configured(test_client, api_key):
    resp = test_client.post("/query", json={"question": "hi"})
    assert resp.status_code == 401


def test_query_rejects_wrong_key(test_client, api_key):
    resp = test_client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": "wrong"}
    )
    assert resp.status_code == 401


def test_query_accepts_valid_key(test_client, api_key):
    resp = test_client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": api_key}
    )
    assert resp.status_code == 200


def test_ingest_requires_key_when_configured(test_client, api_key):
    resp = test_client.post(
        "/ingest", json={"content": "# Doc", "source": "a.md"}
    )
    assert resp.status_code == 401


def test_ingest_accepts_valid_key(test_client, api_key):
    resp = test_client.post(
        "/ingest",
        json={"content": "# Doc", "source": "a.md"},
        headers={"X-API-Key": api_key},
    )
    assert resp.status_code == 200


def test_metrics_requires_key_when_configured(test_client, api_key):
    resp = test_client.get("/metrics")
    assert resp.status_code == 401


def test_health_is_public_even_with_key(test_client, api_key):
    # Health must never require auth so load balancers can probe it.
    resp = test_client.get("/health")
    assert resp.status_code == 200


def test_endpoints_open_when_key_unset(test_client):
    # Default settings.API_KEY == "" -> auth disabled.
    assert test_client.post("/query", json={"question": "hi"}).status_code == 200
    assert test_client.get("/metrics").status_code == 200
