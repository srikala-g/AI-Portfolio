"""Tests that run without any API key — they exercise the pure-Python logic.

Run:  pytest -q   (pythonpath is set to src/ in pyproject.toml)
"""

from credit_risk_analyzer.ingest import _split, chunk_pages


def test_split_short_text_is_one_chunk():
    assert _split("short text", size=900, overlap=150) == ["short text"]


def test_split_long_text_overlaps():
    text = "A" * 2000
    chunks = _split(text, size=900, overlap=150)
    assert len(chunks) >= 2
    assert all(len(c) <= 900 for c in chunks)


def test_chunk_pages_preserves_page_numbers():
    pages = [(1, "alpha " * 300), (2, "beta " * 300)]
    chunks = chunk_pages(pages, source="test.pdf")
    assert any(c.page == 1 for c in chunks)
    assert any(c.page == 2 for c in chunks)
    assert all(c.source == "test.pdf" for c in chunks)
