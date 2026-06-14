"""Tests for configurable settings, focused on CORS origin parsing."""

from app.config import Settings


def test_cors_origins_empty_by_default():
    s = Settings(CORS_ORIGINS="")
    assert s.cors_origins_list == []


def test_cors_origins_single():
    s = Settings(CORS_ORIGINS="https://app.example.com")
    assert s.cors_origins_list == ["https://app.example.com"]


def test_cors_origins_multiple_with_whitespace():
    s = Settings(CORS_ORIGINS="https://a.com, https://b.com ,https://c.com")
    assert s.cors_origins_list == [
        "https://a.com",
        "https://b.com",
        "https://c.com",
    ]


def test_cors_origins_ignores_blank_entries():
    s = Settings(CORS_ORIGINS="https://a.com,, ,https://b.com")
    assert s.cors_origins_list == ["https://a.com", "https://b.com"]
