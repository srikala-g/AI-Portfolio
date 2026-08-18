"""A minimal in-memory vector store using cosine similarity.

For single-document Q&A the corpus is small, so a brute-force numpy search is
plenty fast and has zero extra dependencies. It's also fully transparent, which
makes it easy to explain in an interview. Swap in FAISS or a managed vector DB
(Chroma, pgvector, Pinecone) when you scale to many large documents.
"""

from __future__ import annotations

import numpy as np

from .ingest import Chunk


class VectorStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # (n, d), L2-normalised

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Store chunks with their (already computed) embedding vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed = vectors / norms
        self._matrix = normed if self._matrix is None else np.vstack([self._matrix, normed])
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, similarity) pairs for a query embedding."""
        if self._matrix is None or not self._chunks:
            return []
        q = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        scores = self._matrix @ q  # cosine similarity
        top = np.argsort(scores)[::-1][:k]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._chunks)
