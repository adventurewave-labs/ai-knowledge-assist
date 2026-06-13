"""Tests for the structured JSON logging configuration."""

import json
import logging

from app.logging_config import JsonFormatter, configure_logging


def _make_record(**extra):
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formatter_emits_valid_json():
    record = _make_record()
    payload = json.loads(JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_formatter_includes_extra_fields():
    record = _make_record(event="query", question="What is RAG?")
    payload = json.loads(JsonFormatter().format(record))
    assert payload["event"] == "query"
    assert payload["question"] == "What is RAG?"


def test_configure_logging_installs_json_handler():
    configure_logging("DEBUG")
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logging.getLogger().handlers) == 1
