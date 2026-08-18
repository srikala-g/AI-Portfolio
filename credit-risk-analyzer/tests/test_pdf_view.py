"""Tests for the citation page-viewer's rasterization + disk cache.

Uses the real Apple 10-K fixture (no network/API key involved -- pypdfium2
renders locally) so page numbers line up with the ones asserted elsewhere
(tests/test_metrics.py) against the same fixture.
"""

from pathlib import Path

import pytest

import credit_risk_analyzer.pdf_view as pdf_view

FIXTURE = "tests/fixtures/apple-10k-2025.pdf"


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_view, "PAGE_IMAGE_CACHE_DIR", str(tmp_path / "page-images"))


def test_renders_a_page_to_a_real_png_file():
    path = pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=35)

    assert Path(path).exists()
    assert Path(path).suffix == ".png"
    assert Path(path).stat().st_size > 0


def test_cache_hit_returns_same_path_without_rerendering(monkeypatch):
    first = pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=35)

    def _boom(*a, **k):
        raise AssertionError("cache hit should not touch pdfium.PdfDocument")

    monkeypatch.setattr(pdf_view.pdfium, "PdfDocument", _boom)
    second = pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=35)

    assert first == second


def test_different_doc_ids_do_not_share_a_cache_entry():
    """Same page number, different doc_id -- must not collide (this is the
    exact identity guarantee a citation click depends on: doc_id, not just
    page number, decides which cached image comes back)."""
    path_a = pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=10)
    path_b = pdf_view.get_page_image_path(FIXTURE, doc_id="docB", page=10)

    assert path_a != path_b


def test_out_of_range_page_raises_page_image_error_not_crash():
    with pytest.raises(pdf_view.PageImageError, match="out of range"):
        pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=9999)


def test_zero_or_negative_page_raises_page_image_error():
    with pytest.raises(pdf_view.PageImageError, match="out of range"):
        pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=0)


def test_missing_pdf_file_raises_page_image_error_not_crash():
    with pytest.raises(pdf_view.PageImageError, match="Could not open"):
        pdf_view.get_page_image_path("/no/such/file.pdf", doc_id="docA", page=1)


# ---------------------------------------------------------------------------
# Best-effort value highlighting (Route A: search the page's own text).
# Real values/pages pulled from tests/test_metrics.py's RAW_LINE_ITEMS
# (Apple FY2025 10-K, tests/fixtures/apple-10k-2025.pdf).
# ---------------------------------------------------------------------------


def test_single_match_returns_a_highlighted_image_distinct_from_the_plain_page():
    """cash_and_equivalents (35,934) appears exactly once on page 35 --
    a confident single match, so a highlighted variant is produced."""
    plain_path = pdf_view.get_page_image_path(FIXTURE, doc_id="docA", page=35)
    highlighted_path = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=35, value=35934.0
    )

    assert highlighted_path is not None
    assert Path(highlighted_path).exists()
    assert highlighted_path != plain_path  # a separate cached file, not the plain page


def test_zero_matches_falls_back_to_none():
    """interest_expense_estimate (2,600) is printed on page 29 only as
    prose ("$2.6 billion"), never as the literal table figure "2,600" --
    a real, expected miss, not a bug. Must fall back cleanly, not raise."""
    result = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=29, value=2600.0
    )

    assert result is None


def test_multiple_matches_falls_back_to_none_not_a_guess():
    """total_assets (359,241) appears twice on page 35 -- once as "Total
    assets", once (coincidentally, by the accounting identity assets =
    liabilities + equity) as "Total liabilities and shareholders' equity".
    Per policy, ambiguity falls back to no highlight rather than boxing an
    arbitrary one of the two and presenting it as the source."""
    result = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=35, value=359241.0
    )

    assert result is None


def test_highlight_cache_hit_returns_same_path_without_rerendering(monkeypatch):
    first = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=35, value=35934.0
    )

    def _boom(*a, **k):
        raise AssertionError("cache hit should not touch pdfium.PdfDocument")

    monkeypatch.setattr(pdf_view.pdfium, "PdfDocument", _boom)
    second = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=35, value=35934.0
    )

    assert first == second


def test_different_values_on_the_same_page_do_not_share_a_cache_entry():
    path_cash = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=35, value=35934.0
    )
    path_inventories = pdf_view.get_highlighted_page_image_path(
        FIXTURE, doc_id="docA", page=35, value=5718.0
    )

    assert path_cash is not None
    assert path_inventories is not None
    assert path_cash != path_inventories


def test_highlight_falls_back_cleanly_for_a_nonexistent_pdf_path():
    """Even a broken source path must not raise from the highlight layer --
    it's strictly best-effort; app.show_page relies on falling through to
    its own plain get_page_image_path call (which does raise, appropriately,
    since that's the real page-open, not the highlight convenience layer)."""
    result = pdf_view.get_highlighted_page_image_path(
        "/no/such/file.pdf", doc_id="docA", page=1, value=100.0
    )

    assert result is None
