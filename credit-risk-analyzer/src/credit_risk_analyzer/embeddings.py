"""Local text embeddings via sentence-transformers.

The embedding backend is isolated here so the rest of the app can keep the same
interface while switching to a local model and a disk-backed cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # sentence-transformers is optional until installed
    SentenceTransformer = None

from .config import CACHE_DIR, EMBED_BATCH_SIZE, EMBEDDING_MODEL

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str | None = None) -> Any:
    """Load a sentence-transformers model once per model name."""
    name = model_name or EMBEDDING_MODEL
    if name not in _model_cache:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for local embeddings")
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def _build_cache_key(
    *,
    cache_key: str | bytes | None = None,
    source_bytes: bytes | None = None,
    metadata: list[dict[str, Any]] | None = None,
    model_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> str:
    """Build a stable cache key that changes when the source bytes or config changes."""
    if cache_key is None and source_bytes is None and metadata is None:
        cache_key = ""

    if isinstance(cache_key, bytes):
        cache_key = cache_key.decode("utf-8", errors="ignore")

    payload = {
        "base": cache_key or "",
        "source_bytes": source_bytes.hex() if source_bytes is not None else "",
        "model_name": model_name or EMBEDDING_MODEL,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "metadata": metadata or [],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _cache_path(cache_key: str) -> Path:
    return Path(CACHE_DIR) / f"{cache_key}.npz"


def _load_cached_vectors(cache_key: str) -> np.ndarray | None:
    """Load a cached embedding matrix if it exists and is readable."""
    cache_file = _cache_path(cache_key)
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as data:
            if "vectors" not in data:
                return None
            return np.asarray(data["vectors"], dtype="float32")
    except Exception:
        return None


def _save_cached_vectors(
    cache_key: str,
    vectors: np.ndarray,
    texts: list[str],
    metadata: list[dict[str, Any]] | None,
) -> None:
    """Persist vectors and chunk metadata to disk as an .npz archive."""
    cache_file = _cache_path(cache_key)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    pages = [item.get("page", -1) if isinstance(item, dict) else -1 for item in metadata or []]
    chunk_ids = [
        item.get("chunk_id", -1) if isinstance(item, dict) else -1 for item in metadata or []
    ]
    sources = [item.get("source", "") if isinstance(item, dict) else "" for item in metadata or []]
    metadata_json = [
        json.dumps(item, sort_keys=True, ensure_ascii=False)
        if isinstance(item, dict)
        else json.dumps({"value": str(item)})
        for item in metadata or []
    ]

    np.savez_compressed(
        cache_file,
        vectors=np.asarray(vectors, dtype="float32"),
        texts=np.array(texts, dtype="U"),
        pages=np.array(pages, dtype="int64"),
        chunk_ids=np.array(chunk_ids, dtype="int64"),
        sources=np.array(sources, dtype="U"),
        metadata=np.array(metadata_json, dtype="U"),
    )


def _embed_texts_impl(
    texts: list[str], on_progress=None, model_name: str | None = None
) -> np.ndarray:
    """Embed a list of texts in batches -> (n, d) float32 array."""
    if not texts:
        return np.empty((0, 0), dtype="float32")

    model = _get_model(model_name)
    vectors: list[np.ndarray] = []
    total = len(texts)
    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        batch_vectors = model.encode(
            batch,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.asarray(batch_vectors, dtype="float32")
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        vectors.append(arr)
        if on_progress is not None:
            on_progress(min(start + len(batch), total), total)

    if not vectors:
        return np.empty((0, 0), dtype="float32")
    return np.vstack(vectors)


def embed_texts(
    texts: list[str],
    on_progress=None,
    *,
    cache_key: str | bytes | None = None,
    metadata: list[dict[str, Any]] | None = None,
    model_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    source_bytes: bytes | None = None,
) -> np.ndarray:
    """Embed document chunks for storage/retrieval, using a local model and disk cache."""
    if not texts:
        return np.empty((0, 0), dtype="float32")

    cache_key_name = _build_cache_key(
        cache_key=cache_key,
        source_bytes=source_bytes,
        metadata=metadata,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    cached = _load_cached_vectors(cache_key_name)
    if cached is not None:
        return cached

    vectors = _embed_texts_impl(texts, on_progress=on_progress, model_name=model_name)
    if cache_key is not None or source_bytes is not None:
        _save_cached_vectors(cache_key_name, vectors, texts, metadata)
    return vectors


def embed_query(text: str) -> np.ndarray:
    """Embed a single search query -> (d,) float32 array."""
    return embed_texts([text])[0]
