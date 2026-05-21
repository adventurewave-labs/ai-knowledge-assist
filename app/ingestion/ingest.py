import hashlib
from typing import Any

from app.ingestion.markdown_parser import MarkdownParser
from app.models.schemas import IngestResponse
from app.rag.retriever import VectorStoreRetriever


class IngestPipeline:
    def __init__(self, retriever: VectorStoreRetriever) -> None:
        self.retriever = retriever
        self.parser = MarkdownParser()
        # Track ingested source+hash pairs to avoid re-ingesting identical content
        self._seen: set[str] = set()

    def ingest_document(
        self,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> IngestResponse:
        metadata = metadata or {}

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        dedup_key = f"{source}::{content_hash}"

        if dedup_key in self._seen:
            existing_count = self.retriever.get_document_count_for_source(source)
            return IngestResponse(
                document_id=content_hash,
                chunks_created=existing_count,
                source=source,
            )

        documents = self.parser.parse(content, source)

        for doc in documents:
            doc.metadata.update(metadata)
            doc.metadata["content_hash"] = content_hash

        if documents:
            self.retriever.add_documents(documents)

        self._seen.add(dedup_key)

        return IngestResponse(
            document_id=content_hash,
            chunks_created=len(documents),
            source=source,
        )
