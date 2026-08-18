"""The RAG pipeline: build an index from a PDF, then answer questions from it.

Flow:  ingest -> embed -> store   (build_index)
       question -> embed -> retrieve top-k -> prompt LLM -> grounded answer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from .config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_K,
)
from .embeddings import embed_query, embed_texts
from .ingest import Chunk, ingest_pdf
from .retry import call_with_retry
from .vectorstore import VectorStore

_client = None


def _get_client() -> genai.Client:
    """Create the client on first use so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


@dataclass
class Answer:
    text: str
    sources: list[tuple[Chunk, float]]  # the chunks that were retrieved


def build_index(pdf_path: str, on_progress=None) -> VectorStore:
    """Ingest a PDF and return a searchable vector store.

    ``on_progress(done, total)`` is forwarded to the embedding step so callers
    can display indexing progress.
    """
    chunks = ingest_pdf(pdf_path)
    if not chunks:
        raise ValueError("No extractable text found. Is this a scanned PDF? (OCR needed.)")

    source_bytes = Path(pdf_path).read_bytes()
    metadata = [
        {"text": chunk.text, "page": chunk.page, "chunk_id": chunk.chunk_id, "source": chunk.source}
        for chunk in chunks
    ]
    vectors = embed_texts(
        [chunk.text for chunk in chunks],
        on_progress=on_progress,
        metadata=metadata,
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        source_bytes=source_bytes,
    )
    store = VectorStore()
    store.add(chunks, vectors)
    return store


def _format_context(hits: list[tuple[Chunk, float]]) -> str:
    """Lay out retrieved chunks with page labels the model can cite."""
    blocks = []
    for chunk, _score in hits:
        blocks.append(f"[p. {chunk.page}] {chunk.text}")
    return "\n\n---\n\n".join(blocks)


def answer_question(store: VectorStore, question: str, k: int = TOP_K) -> Answer:
    """Retrieve relevant context and produce a grounded, cited answer."""
    hits = store.search(embed_query(question), k=k)
    context = _format_context(hits)

    user_prompt = (
        f"Context excerpts from the document:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above and cite page numbers."
    )

    resp = call_with_retry(
        lambda: _get_client().models.generate_content(
            model=CHAT_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
            ),
        )
    )
    return Answer(text=resp.text.strip(), sources=hits)
