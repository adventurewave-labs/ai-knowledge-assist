from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever as LCRetriever

from app.config import settings


class VectorStoreRetriever:
    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._client = chromadb.Client() if settings.CHROMA_PERSIST_DIR == ":memory:" else chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._vectorstore = Chroma(
            client=self._client,
            collection_name="aria_documents",
            embedding_function=self._embeddings,
        )
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_retriever(
        self,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> LCRetriever:
        k = top_k or settings.TOP_K
        search_kwargs: dict[str, Any] = {"k": k}
        if filters:
            search_kwargs["filter"] = filters
        return self._vectorstore.as_retriever(search_kwargs=search_kwargs)

    def add_documents(self, documents: list[Document]) -> None:
        self._vectorstore.add_documents(documents)

    def get_document_count(self) -> int:
        try:
            return self._vectorstore._collection.count()
        except Exception:
            return 0

    def get_document_count_for_source(self, source: str) -> int:
        try:
            results = self._vectorstore._collection.get(
                where={"source": source}
            )
            return len(results.get("ids", []))
        except Exception:
            return 0
