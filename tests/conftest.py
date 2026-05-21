from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

SAMPLE_MARKDOWN_SIMPLE = """# Introduction

This is the introduction section.

## Getting Started

Follow these steps to get started with the system.

## Configuration

Set up your environment variables before running.
"""

SAMPLE_MARKDOWN_WITH_FRONTMATTER = """---
title: Test Document
tags: [python, testing]
date: 2024-01-15
---

# Test Document

This document has frontmatter metadata.

## Section One

Content for section one.

## Section Two

Content for section two with a code block:

```python
def hello():
    return "world"
```

More text after the code block.
"""

SAMPLE_MARKDOWN_LONG = (
    "# Long Document\n\n"
    + ("This is a sentence that adds length. " * 20 + "\n\n") * 5
)

SAMPLE_MARKDOWN_TABLE = """# Data Reference

## Supported Formats

| Format | Extension | Notes          |
|--------|-----------|----------------|
| JSON   | .json     | Structured data|
| YAML   | .yaml     | Config files   |
| CSV    | .csv      | Tabular data   |

The table above lists supported formats.
"""

MOCK_LLM_ANSWER = "This is a mocked LLM answer for testing purposes."


@pytest.fixture
def sample_markdown_simple() -> str:
    return SAMPLE_MARKDOWN_SIMPLE


@pytest.fixture
def sample_markdown_with_frontmatter() -> str:
    return SAMPLE_MARKDOWN_WITH_FRONTMATTER


@pytest.fixture
def sample_markdown_long() -> str:
    return SAMPLE_MARKDOWN_LONG


@pytest.fixture
def sample_markdown_table() -> str:
    return SAMPLE_MARKDOWN_TABLE


@pytest.fixture
def mock_llm_response() -> str:
    return MOCK_LLM_ANSWER


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Python is a high-level programming language.",
            metadata={"source": "test.md", "header": "Python", "chunk_index": 0},
        ),
        Document(
            page_content="FastAPI is a modern web framework for Python.",
            metadata={"source": "test.md", "header": "FastAPI", "chunk_index": 0},
        ),
    ]


@pytest.fixture
def mock_retriever(sample_documents):
    retriever = MagicMock()
    retriever.is_ready = True
    retriever.get_document_count.return_value = len(sample_documents)
    retriever.get_document_count_for_source.return_value = 1

    lc_retriever = MagicMock()
    lc_retriever.invoke.return_value = sample_documents
    retriever.get_retriever.return_value = lc_retriever
    retriever.add_documents.return_value = None
    return retriever


@pytest.fixture
def test_client(mock_retriever):
    from app.ingestion.ingest import IngestPipeline
    from app.rag.chain import RAGChain

    mock_chain = MagicMock(spec=RAGChain)
    mock_chain.total_queries = 0
    mock_chain.avg_latency_ms = 0.0
    mock_chain.fallback_count = 0
    mock_chain.query.return_value = MagicMock(
        answer=MOCK_LLM_ANSWER,
        sources=[],
        model_used="gpt-4o-mini",
        latency_ms=42.0,
        model_dump=lambda: {
            "answer": MOCK_LLM_ANSWER,
            "sources": [],
            "model_used": "gpt-4o-mini",
            "latency_ms": 42.0,
        },
    )

    mock_pipeline = MagicMock(spec=IngestPipeline)
    mock_pipeline.ingest_document.return_value = MagicMock(
        document_id="abc123",
        chunks_created=3,
        source="test.md",
        model_dump=lambda: {
            "document_id": "abc123",
            "chunks_created": 3,
            "source": "test.md",
        },
    )

    with patch("app.main.VectorStoreRetriever", return_value=mock_retriever), patch(
        "app.main.RAGChain", return_value=mock_chain
    ), patch("app.main.IngestPipeline", return_value=mock_pipeline):
        from app.main import app

        with TestClient(app) as client:
            yield client
