import numpy as np

import credit_risk_analyzer.embeddings as embeddings_mod


class FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True):
        values = [float(len(text) + ord(self.model_name[-1])) for text in texts]
        return np.array([values], dtype="float32").repeat(len(texts), axis=0)


def test_embedding_cache_round_trip(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(embeddings_mod, "CACHE_DIR", str(cache_dir))

    calls = []

    def fake_get_model(model_name=None):
        calls.append(model_name)
        return FakeModel(model_name or "test-model")

    monkeypatch.setattr(embeddings_mod, "_get_model", fake_get_model)

    metadata = [{"text": "alpha", "page": 1, "chunk_id": 0, "source": "sample.pdf"}]
    first = embeddings_mod.embed_texts(
        ["alpha"],
        cache_key="round-trip",
        metadata=metadata,
        model_name="test-model",
        chunk_size=100,
        chunk_overlap=20,
    )
    second = embeddings_mod.embed_texts(
        ["alpha"],
        cache_key="round-trip",
        metadata=metadata,
        model_name="test-model",
        chunk_size=100,
        chunk_overlap=20,
    )

    np.testing.assert_allclose(first, second)
    assert len(calls) == 1
    assert any(cache_dir.glob("*.npz"))


def test_embedding_cache_invalidation_on_model_change(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(embeddings_mod, "CACHE_DIR", str(cache_dir))

    calls = []

    def fake_get_model(model_name=None):
        calls.append(model_name)
        return FakeModel(model_name or "test-model")

    monkeypatch.setattr(embeddings_mod, "_get_model", fake_get_model)

    metadata = [{"text": "alpha", "page": 1, "chunk_id": 0, "source": "sample.pdf"}]
    first = embeddings_mod.embed_texts(
        ["alpha"],
        cache_key="invalidate",
        metadata=metadata,
        model_name="model-a",
        chunk_size=100,
        chunk_overlap=20,
    )
    second = embeddings_mod.embed_texts(
        ["alpha"],
        cache_key="invalidate",
        metadata=metadata,
        model_name="model-b",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(calls) == 2
    assert not np.allclose(first, second)
