from pathlib import Path

import pytest

from app.ingestion.markdown_parser import MarkdownParser


@pytest.fixture
def parser():
    return MarkdownParser(chunk_size=512, chunk_overlap=64)


def test_parse_simple_returns_documents(parser, sample_markdown_simple):
    docs = parser.parse(sample_markdown_simple, source="test.md")
    assert len(docs) >= 1
    for doc in docs:
        assert doc.page_content.strip()
        assert doc.metadata["source"] == "test.md"


def test_parse_frontmatter_extracts_metadata(parser, sample_markdown_with_frontmatter):
    docs = parser.parse(sample_markdown_with_frontmatter, source="fm.md")
    assert len(docs) >= 1
    first = docs[0]
    assert first.metadata.get("title") == "Test Document"
    assert "python" in first.metadata.get("tags", [])


def test_parse_headers_create_sections(parser, sample_markdown_simple):
    docs = parser.parse(sample_markdown_simple, source="sections.md")
    headers = [d.metadata.get("header") for d in docs if d.metadata.get("header")]
    assert len(headers) >= 2


def test_parse_header_level_recorded(parser, sample_markdown_simple):
    docs = parser.parse(sample_markdown_simple, source="levels.md")
    for doc in docs:
        if doc.metadata.get("header"):
            assert doc.metadata["header_level"] in (1, 2, 3, 4)


def test_code_block_not_split(parser, sample_markdown_with_frontmatter):
    docs = parser.parse(sample_markdown_with_frontmatter, source="code.md")
    full_text = "\n".join(d.page_content for d in docs)
    # The code block must appear intact somewhere
    assert 'def hello():' in full_text
    assert 'return "world"' in full_text


def test_chunk_index_present(parser, sample_markdown_simple):
    docs = parser.parse(sample_markdown_simple, source="idx.md")
    for doc in docs:
        assert "chunk_index" in doc.metadata


def test_long_document_produces_multiple_chunks(parser, sample_markdown_long):
    docs = parser.parse(sample_markdown_long, source="long.md")
    assert len(docs) >= 2


def test_no_empty_chunks(parser, sample_markdown_simple):
    docs = parser.parse(sample_markdown_simple, source="empty.md")
    for doc in docs:
        assert doc.page_content.strip(), "Empty chunk found"


def test_table_preserved(parser, sample_markdown_table):
    docs = parser.parse(sample_markdown_table, source="table.md")
    full_text = "\n".join(d.page_content for d in docs)
    assert "JSON" in full_text
    assert "YAML" in full_text


def test_parse_file(parser, tmp_path, sample_markdown_with_frontmatter):
    md_file = tmp_path / "doc.md"
    md_file.write_text(sample_markdown_with_frontmatter, encoding="utf-8")
    docs = parser.parse_file(md_file)
    assert len(docs) >= 1
    assert docs[0].metadata["source"] == str(md_file)


def test_custom_chunk_size_respected():
    small_parser = MarkdownParser(chunk_size=100, chunk_overlap=10)
    long_section = "# Big Section\n\n" + ("Word " * 200)
    docs = small_parser.parse(long_section, source="big.md")
    for doc in docs:
        assert len(doc.page_content) <= 200, "Chunk significantly exceeds chunk_size"


def test_document_without_frontmatter(parser):
    md = "# Title\n\nSome content here."
    docs = parser.parse(md, source="no_fm.md")
    assert len(docs) >= 1
    assert "title" not in docs[0].metadata
