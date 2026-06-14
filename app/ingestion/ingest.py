import hashlib
import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingestion.markdown_parser import MarkdownParser
from app.logging_config import get_logger
from app.models.schemas import IngestResponse
from app.rag.retriever import VectorStoreRetriever

logger = get_logger(__name__)


class IngestPipeline:
    def __init__(
        self,
        retriever: VectorStoreRetriever,
        dedup_store_path: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.parser = MarkdownParser()
        # Track ingested source+hash pairs to avoid re-ingesting identical
        # content. Persisted to disk so dedup state survives process restarts.
        self._dedup_store_path = (
            dedup_store_path
            if dedup_store_path is not None
            else settings.DEDUP_STORE_PATH
        )
        self._seen: set[str] = self._load_seen()

    def _load_seen(self) -> set[str]:
        """Load previously persisted dedup keys, if a store path is configured."""
        if not self._dedup_store_path:
            return set()
        path = Path(self._dedup_store_path)
        if not path.exists():
            return set()
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            keys = data.get("seen", []) if isinstance(data, dict) else list(data)
            return set(keys)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "failed to load dedup store; starting empty",
                extra={"event": "dedup_load_error", "error": str(exc)},
            )
            return set()

    def _persist_seen(self) -> None:
        """Persist the current dedup key set to disk if a store path is set."""
        if not self._dedup_store_path:
            return
        path = Path(self._dedup_store_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump({"seen": sorted(self._seen)}, f)
        except OSError as exc:
            logger.warning(
                "failed to persist dedup store",
                extra={"event": "dedup_persist_error", "error": str(exc)},
            )

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
            logger.info(
                "ingest skipped (duplicate)",
                extra={
                    "event": "ingest_duplicate",
                    "source": source,
                    "document_id": content_hash,
                },
            )
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
        self._persist_seen()

        logger.info(
            "ingest completed",
            extra={
                "event": "ingest",
                "source": source,
                "document_id": content_hash,
                "chunks_created": len(documents),
            },
        )

        return IngestResponse(
            document_id=content_hash,
            chunks_created=len(documents),
            source=source,
        )
