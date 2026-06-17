import pytest

from app.config import settings


@pytest.fixture
def low_limit(monkeypatch):
    """Force a tiny rate limit and start from a clean bucket."""
    from app.rate_limit import limiter

    monkeypatch.setattr(settings, "RATE_LIMIT_RPM", 2)
    limiter.reset()
    yield
    limiter.reset()


def test_429_returned_when_limit_exceeded(test_client, low_limit):
    headers = {"X-API-Key": "ratelimit-429-key"}
    payload = {"question": "hello"}
    statuses = [
        test_client.post("/query", json=payload, headers=headers).status_code
        for _ in range(5)
    ]
    assert 429 in statuses, f"expected a 429 in {statuses}"


def test_retry_after_header_present(test_client, low_limit):
    headers = {"X-API-Key": "ratelimit-retry-key"}
    payload = {"question": "hello"}
    last = None
    for _ in range(6):
        last = test_client.post("/query", json=payload, headers=headers)
        if last.status_code == 429:
            break
    assert last.status_code == 429
    assert last.headers.get("Retry-After") == "60"


def test_health_is_not_rate_limited(test_client, low_limit):
    # Far more requests than the configured limit; /health must never 429.
    statuses = [test_client.get("/health").status_code for _ in range(10)]
    assert all(s == 200 for s in statuses)


def test_metrics_is_not_rate_limited(test_client, low_limit):
    statuses = [test_client.get("/metrics").status_code for _ in range(10)]
    assert all(s == 200 for s in statuses)
