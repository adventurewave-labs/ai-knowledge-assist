"""Structured JSON logging configuration for Project ARIA.

Provides a stdlib-only JSON formatter so the application emits machine-parseable
log lines suitable for aggregation in production. Any field passed via the
logging ``extra={...}`` argument is merged into the emitted JSON object.
"""

import datetime
import json
import logging
import sys
from typing import Any

# Attributes that exist on every LogRecord; everything else passed via ``extra``
# is treated as a structured field and included in the JSON payload.
_RESERVED_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(
                record.created, tz=datetime.timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any structured fields supplied through ``extra``.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging to emit structured JSON to stdout.

    Idempotent: replaces any existing handlers so repeated calls (e.g. across
    app restarts within the same process) do not duplicate log lines.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Tame noisy third-party loggers while keeping our app logs verbose.
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.access").propagate = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger; thin wrapper for consistent imports."""
    return logging.getLogger(name)
