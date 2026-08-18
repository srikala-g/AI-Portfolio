"""Tests for app.show_page: (doc_id, page, doc_paths) -> rasterized page image.

Uses the real Apple 10-K fixture so pypdfium2 actually renders (no network
involved), isolated to a temp page-image cache dir per test.
"""

from pathlib import Path

import pytest

import app
import credit_risk_analyzer.pdf_view as pdf_view

FIXTURE = "tests/fixtures/apple-10k-2025.pdf"


@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_view, "PAGE_IMAGE_CACHE_DIR", str(tmp_path / "page-images"))


def test_show_page_resolves_doc_id_to_the_correct_pdf_and_page():
    doc_paths = {"doc-a": FIXTURE}

    image_path, caption, group_update = app.show_page("doc-a", 35, doc_paths)

    assert image_path is not None
    assert Path(image_path).exists()
    assert "apple-10k-2025.pdf" in caption
    assert "35" in caption
    assert group_update.visible is True


def test_show_page_unknown_doc_id_shows_error_not_crash():
    """A citation whose document is not (or no longer) in doc_paths -- e.g.
    from a stale session -- must show a message, never raise or silently
    open the wrong file."""
    image_path, caption, group_update = app.show_page("does-not-exist", 10, {})

    assert image_path is None
    assert "Could not find the source document" in caption
    assert group_update.visible is True


def test_show_page_out_of_range_page_shows_error_not_crash():
    doc_paths = {"doc-a": FIXTURE}

    image_path, caption, group_update = app.show_page("doc-a", 9999, doc_paths)

    assert image_path is None
    assert "Could not open page" in caption
    assert group_update.visible is True


def test_show_page_missing_doc_id_or_page_hides_viewer():
    image_path, caption, group_update = app.show_page(None, None, {})

    assert image_path is None
    assert group_update.visible is False


def test_show_page_two_doc_ids_never_cross_resolve():
    """The correctness guarantee this whole feature depends on: doc_id
    selects which file gets opened, not the page number or upload order."""
    doc_paths = {"doc-a": FIXTURE, "doc-b": "/no/such/other/file.pdf"}

    image_path_a, caption_a, _ = app.show_page("doc-a", 35, doc_paths)
    image_path_b, caption_b, _ = app.show_page("doc-b", 35, doc_paths)

    assert image_path_a is not None
    assert "apple-10k-2025.pdf" in caption_a
    assert image_path_b is None
    assert "Could not open page" in caption_b


# ---------------------------------------------------------------------------
# Best-effort value highlighting, threaded through show_page's `value` param
# (the metrics dashboard's citation links; Q&A's never pass one).
# ---------------------------------------------------------------------------


def test_show_page_with_a_confidently_matched_value_returns_a_page_still():
    """A single-match value (cash_and_equivalents, 35,934 on page 35) still
    opens the page correctly -- the caption/visibility contract is
    unaffected by whether a highlight was found."""
    doc_paths = {"doc-a": FIXTURE}

    image_path, caption, group_update = app.show_page("doc-a", 35, doc_paths, value=35934.0)

    assert image_path is not None
    assert Path(image_path).exists()
    assert "apple-10k-2025.pdf" in caption
    assert group_update.visible is True


def test_show_page_falls_back_to_plain_page_when_value_has_no_confident_match():
    """A value that can't be pinned down (interest_expense_estimate, printed
    only as "$2.6 billion" prose on page 29) must still open the plain page
    -- exactly the same result as calling show_page without a value at all."""
    doc_paths = {"doc-a": FIXTURE}

    with_value = app.show_page("doc-a", 29, doc_paths, value=2600.0)
    without_value = app.show_page("doc-a", 29, doc_paths)

    # gr.Group is a fresh object each call (no __eq__), so compare the
    # meaningful fields rather than the raw tuples.
    assert with_value[0] is not None  # still opens
    assert with_value[0] == without_value[0]  # same image path: no highlight attempted survives
    assert with_value[1] == without_value[1]  # same caption
    assert with_value[2].visible is True
    assert without_value[2].visible is True


def test_show_page_falls_back_to_plain_page_on_ambiguous_multi_match_value():
    """total_assets (359,241) appears twice on page 35 -- ambiguous, so the
    page still opens but without a highlight, same as the no-value case."""
    doc_paths = {"doc-a": FIXTURE}

    with_value = app.show_page("doc-a", 35, doc_paths, value=359241.0)
    without_value = app.show_page("doc-a", 35, doc_paths)

    assert with_value[0] is not None
    assert with_value[0] == without_value[0]
    assert with_value[1] == without_value[1]
    assert with_value[2].visible is True
    assert without_value[2].visible is True


def test_show_page_none_value_behaves_exactly_like_no_value_argument():
    doc_paths = {"doc-a": FIXTURE}

    explicit_none = app.show_page("doc-a", 35, doc_paths, value=None)
    omitted = app.show_page("doc-a", 35, doc_paths)

    assert explicit_none[0] == omitted[0]
    assert explicit_none[1] == omitted[1]
    assert explicit_none[2].visible == omitted[2].visible


def test_on_metrics_citation_click_reads_doc_page_and_value_from_event_payload():
    """Adapter test: evt.doc/evt.page/evt.value (as delivered by the
    dashboard's js_on_load trigger payload -- see METRICS_JS_ON_LOAD) route
    correctly into show_page, producing a highlighted result for a
    confident single match."""

    class _FakeEvent:
        doc = "doc-a"
        page = "35"
        value = "35934"

    doc_paths = {"doc-a": FIXTURE}
    image_path, caption, group_update = app.on_metrics_citation_click(doc_paths, _FakeEvent())

    assert image_path is not None
    assert group_update.visible is True


def test_on_metrics_citation_click_handles_a_link_with_no_data_value():
    """A citation link with no data-value attribute (Citation.value was
    None) means evt.value is simply absent from the payload -- the adapter
    must not raise AttributeError, and the page should still open plainly."""

    class _FakeEventNoValue:
        doc = "doc-a"
        page = "35"
        # no `value` attribute at all -- matches EventData.__getattr__
        # raising AttributeError for a key that was never in the payload.

    doc_paths = {"doc-a": FIXTURE}
    image_path, caption, group_update = app.on_metrics_citation_click(
        doc_paths, _FakeEventNoValue()
    )

    assert image_path is not None
    assert group_update.visible is True
