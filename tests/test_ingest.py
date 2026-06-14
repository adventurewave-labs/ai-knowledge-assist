from unittest.mock import MagicMock

import pytest

from app.ingestion.ingest import IngestPipeline


@pytest.fixture
def pipeline(mock_retriever, tmp_path):
    # Use a per-test dedup store so persistence does not leak across tests.
    return IngestPipeline(
        retriever=mock_retriever,
        dedup_store_path=str(tmp_path / "dedup.json"),
    )


def test_ingest_returns_response(pipeline, sample_markdown_simple):
    resp = pipeline.ingest_document(
        content=sample_markdown_simple,
        source="test.md",
    )
    assert resp.source == "test.md"
    assert resp.chunks_created >= 1
    assert len(resp.document_id) > 0


def test_ingest_calls_add_documents(pipeline, mock_retriever, sample_markdown_simple):
    pipeline.ingest_document(content=sample_markdown_simple, source="test.md")
    mock_retriever.add_documents.assert_called_once()


def test_ingest_deduplication(pipeline, mock_retriever, sample_markdown_simple):
    pipeline.ingest_document(content=sample_markdown_simple, source="test.md")
    pipeline.ingest_document(content=sample_markdown_simple, source="test.md")
    # add_documents should only be called once due to deduplication
    assert mock_retriever.add_documents.call_count == 1


def test_ingest_different_sources_both_indexed(pipeline, mock_retriever, sample_markdown_simple):
    pipeline.ingest_document(content=sample_markdown_simple, source="doc_a.md")
    pipeline.ingest_document(content=sample_markdown_simple, source="doc_b.md")
    assert mock_retriever.add_documents.call_count == 2


def test_ingest_metadata_merged(pipeline, mock_retriever, sample_markdown_simple):
    extra = {"project": "aria", "version": "1.0"}
    pipeline.ingest_document(
        content=sample_markdown_simple, source="meta.md", metadata=extra
    )
    call_args = mock_retriever.add_documents.call_args[0][0]
    for doc in call_args:
        assert doc.metadata.get("project") == "aria"


def test_ingest_document_id_is_hash(pipeline, sample_markdown_simple):
    resp = pipeline.ingest_document(content=sample_markdown_simple, source="hash.md")
    assert len(resp.document_id) == 16


def test_ingest_with_frontmatter(pipeline, mock_retriever, sample_markdown_with_frontmatter):
    resp = pipeline.ingest_document(
        content=sample_markdown_with_frontmatter, source="fm.md"
    )
    assert resp.chunks_created >= 1


def test_dedup_persists_across_restart(mock_retriever, tmp_path, sample_markdown_simple):
    store = str(tmp_path / "dedup.json")

    first = IngestPipeline(retriever=mock_retriever, dedup_store_path=store)
    first.ingest_document(content=sample_markdown_simple, source="persist.md")
    assert mock_retriever.add_documents.call_count == 1

    # A fresh pipeline pointed at the same store should load prior dedup keys
    # and skip re-ingesting identical content.
    second = IngestPipeline(retriever=mock_retriever, dedup_store_path=store)
    second.ingest_document(content=sample_markdown_simple, source="persist.md")
    assert mock_retriever.add_documents.call_count == 1


def test_dedup_store_file_written(mock_retriever, tmp_path, sample_markdown_simple):
    store = tmp_path / "dedup.json"
    pipe = IngestPipeline(retriever=mock_retriever, dedup_store_path=str(store))
    pipe.ingest_document(content=sample_markdown_simple, source="written.md")
    assert store.exists()


def test_dedup_in_memory_when_no_store(mock_retriever, sample_markdown_simple):
    # Empty path disables persistence; dedup still works within the instance.
    pipe = IngestPipeline(retriever=mock_retriever, dedup_store_path="")
    pipe.ingest_document(content=sample_markdown_simple, source="mem.md")
    pipe.ingest_document(content=sample_markdown_simple, source="mem.md")
    assert mock_retriever.add_documents.call_count == 1
