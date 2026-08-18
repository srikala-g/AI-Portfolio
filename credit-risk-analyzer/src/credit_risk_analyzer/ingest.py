"""Turn a PDF into clean, page-tagged text chunks ready for embedding.

Pipeline:  PDF file -> per-page text -> overlapping chunks (keeping the page
number so we can cite it later).
"""

from __future__ import annotations

from dataclasses import dataclass

from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    """A single retrievable piece of the document."""

    text: str
    page: int  # 1-based page number, used for citations
    chunk_id: int
    source: str  # file name, so a multi-doc version can attribute results


def load_pdf_pages(path: str) -> list[tuple[int, str]]:
    """Return a list of (page_number, page_text) for a PDF."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Character-based sliding window. Simple, transparent, easy to explain."""
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    step = max(size - overlap, 1)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def chunk_pages(pages: list[tuple[int, str]], source: str) -> list[Chunk]:
    """Split each page into overlapping chunks, preserving the page number."""
    chunks: list[Chunk] = []
    cid = 0
    for page_num, page_text in pages:
        for piece in _split(page_text, CHUNK_SIZE, CHUNK_OVERLAP):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page_num, chunk_id=cid, source=source))
                cid += 1
    return chunks


def ingest_pdf(path: str, source: str | None = None) -> list[Chunk]:
    """Convenience wrapper: file path -> list of Chunk."""
    import os

    source = source or os.path.basename(path)
    return chunk_pages(load_pdf_pages(path), source)
