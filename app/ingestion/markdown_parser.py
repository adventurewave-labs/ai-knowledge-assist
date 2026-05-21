import re
from pathlib import Path
from typing import Any

import frontmatter
from langchain_core.documents import Document

from app.config import settings


_HEADER_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
_FENCED_CODE_RE = re.compile(r"(```[\w]*\n.*?```)", re.DOTALL)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


class MarkdownParser:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    def parse(self, content: str, source: str) -> list[Document]:
        post = frontmatter.loads(content)
        body: str = post.content
        fm_meta: dict[str, Any] = dict(post.metadata)

        base_metadata: dict[str, Any] = {"source": source, **fm_meta}

        sections = self._split_by_headers(body)

        documents: list[Document] = []
        for section_text, section_meta in sections:
            merged_meta = {**base_metadata, **section_meta}
            chunks = self._chunk_text(section_text)
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                chunk_meta = {**merged_meta, "chunk_index": i}
                documents.append(Document(page_content=chunk, metadata=chunk_meta))

        return documents

    def parse_file(self, path: Path) -> list[Document]:
        content = path.read_text(encoding="utf-8")
        return self.parse(content, source=str(path))

    def _split_by_headers(
        self, body: str
    ) -> list[tuple[str, dict[str, Any]]]:
        # Protect code blocks from being split by header detection
        protected, placeholders = self._protect_code_blocks(body)

        matches = list(_HEADER_RE.finditer(protected))
        if not matches:
            restored = self._restore_code_blocks(protected, placeholders)
            return [(restored, {})]

        sections: list[tuple[str, dict[str, Any]]] = []

        # Text before the first header
        preamble = protected[: matches[0].start()].strip()
        if preamble:
            restored = self._restore_code_blocks(preamble, placeholders)
            sections.append((restored, {}))

        for idx, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(protected)
            section_body = protected[start:end].strip()
            restored = self._restore_code_blocks(section_body, placeholders)
            meta: dict[str, Any] = {
                "header": title,
                "header_level": level,
            }
            sections.append((restored, meta))

        return sections

    def _protect_code_blocks(
        self, text: str
    ) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        counter = [0]

        def replacer(m: re.Match) -> str:
            key = f"__CODE_BLOCK_{counter[0]}__"
            placeholders[key] = m.group(0)
            counter[0] += 1
            return key

        protected = _FENCED_CODE_RE.sub(replacer, text)
        return protected, placeholders

    def _restore_code_blocks(
        self, text: str, placeholders: dict[str, str]
    ) -> str:
        for key, original in placeholders.items():
            text = text.replace(key, original)
        return text

    def _chunk_text(self, text: str) -> list[str]:
        if not text.strip():
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Keep code blocks intact even during chunking
        protected, placeholders = self._protect_code_blocks(text)

        # Split on paragraph boundaries first, then sentence boundaries
        paragraphs = re.split(r"\n{2,}", protected)

        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            if current_len + para_len + 1 <= self.chunk_size:
                current_parts.append(para)
                current_len += para_len + 1
            else:
                if current_parts:
                    chunk = "\n\n".join(current_parts)
                    chunks.append(self._restore_code_blocks(chunk, placeholders))

                # Overlap: carry forward up to chunk_overlap characters from the last part
                if current_parts and self.chunk_overlap > 0:
                    overlap_text = current_parts[-1]
                    if len(overlap_text) > self.chunk_overlap:
                        # Find the last sentence boundary within chunk_overlap characters
                        tail = overlap_text[-self.chunk_overlap :]
                        boundary = tail.rfind(". ")
                        if boundary != -1:
                            overlap_text = tail[boundary + 2 :]
                        else:
                            overlap_text = tail
                    current_parts = [overlap_text]
                    current_len = len(overlap_text)
                else:
                    current_parts = []
                    current_len = 0

                if para_len > self.chunk_size:
                    # Para is too large even by itself — split at sentence boundaries
                    sentence_chunks = self._split_large_paragraph(para)
                    for sc in sentence_chunks[:-1]:
                        chunks.append(
                            self._restore_code_blocks(sc, placeholders)
                        )
                    # Last sentence chunk becomes the start of the next accumulation
                    if sentence_chunks:
                        current_parts = [sentence_chunks[-1]]
                        current_len = len(sentence_chunks[-1])
                else:
                    current_parts.append(para)
                    current_len += para_len + 1

        if current_parts:
            chunk = "\n\n".join(current_parts)
            chunks.append(self._restore_code_blocks(chunk, placeholders))

        return chunks if chunks else [text]

    def _split_large_paragraph(self, text: str) -> list[str]:
        sentences = _SENTENCE_END_RE.split(text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len > self.chunk_size:
                if current_parts:
                    chunks.append(" ".join(current_parts))
                    current_parts = []
                    current_len = 0
                word_chunks = self._split_by_words(sentence)
                chunks.extend(word_chunks[:-1])
                if word_chunks:
                    current_parts = [word_chunks[-1]]
                    current_len = len(word_chunks[-1])
            elif current_len + sentence_len + 1 <= self.chunk_size:
                current_parts.append(sentence)
                current_len += sentence_len + 1
            else:
                if current_parts:
                    chunks.append(" ".join(current_parts))
                current_parts = [sentence]
                current_len = sentence_len

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks if chunks else [text]

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0
        for word in words:
            wlen = len(word)
            if current_len + wlen + 1 <= self.chunk_size:
                current_words.append(word)
                current_len += wlen + 1
            else:
                if current_words:
                    chunks.append(" ".join(current_words))
                current_words = [word]
                current_len = wlen
        if current_words:
            chunks.append(" ".join(current_words))
        return chunks if chunks else [text]
