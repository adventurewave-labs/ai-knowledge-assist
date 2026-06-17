def test_root_returns_html(test_client):
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_static_index_accessible(test_client):
    resp = test_client.get("/static/index.html")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_html_contains_expected_elements(test_client):
    html = test_client.get("/").text
    assert "EventSource" in html
    assert "ARIA" in html
    assert "v0.2.0" in html
    assert "/query/stream" in html
    assert "/ingest" in html


def test_html_does_not_persist_api_key_to_localstorage(test_client):
    html = test_client.get("/").text
    # API key must be kept in memory only.
    assert "localStorage" not in html
