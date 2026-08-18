"""Tests for app.py's upload-confirmation and session-reset wiring.

These call the plain handler functions directly (no running Gradio server)
and mock the network-touching calls (build_index, extract_line_items) the
same way tests/test_metrics.py mocks the Gemini client, so these run with no
API key or network access.

app._document_id hashes file content (see app.py), so any test path that
reaches it needs a real file on disk with real bytes -- a bare string like
"10k.pdf" no longer stands in for a path. Tests that never reach
_document_id (the "stage pending upload" and "cancel" branches, which return
before _load_document is called) keep using bare strings, since no file is
ever opened on those branches.
"""

import hashlib
import pathlib

import gradio as gr

import app
from credit_risk_analyzer.vectorstore import VectorStore


class _FakeFinancials:
    def __init__(self, fiscal_years):
        self.fiscal_years = fiscal_years


def _mock_extraction(monkeypatch, fiscal_year=2025, html="<div>dashboard</div>"):
    """Stub out the metrics-extraction pipeline so process_metrics needs no API key."""
    monkeypatch.setattr(
        app, "extract_line_items", lambda path, on_retry=None: _FakeFinancials([fiscal_year])
    )
    monkeypatch.setattr(app, "compute_all_derived_metrics", lambda *a, **k: object())
    monkeypatch.setattr(app, "render_dashboard_html", lambda *a, **k: html)
    monkeypatch.setattr(app, "debt_maturity_chart_data", lambda *a, **k: None)


def _mock_build_index(monkeypatch):
    monkeypatch.setattr(app, "build_index", lambda path, on_progress=None: VectorStore())


def _write_fixture(dir_path, name: str, content: bytes) -> str:
    """Write a small real file with real bytes -- app._document_id hashes content."""
    p = pathlib.Path(dir_path) / name
    p.write_bytes(content)
    return str(p)


def _expected_doc_id(content: bytes) -> str:
    """Computed independently of app._document_id, so a test asserting against
    this is actually checking the hash-of-content behavior, not just echoing
    whatever the implementation happens to do."""
    return hashlib.sha256(content).hexdigest()


DOC_A_CONTENT = b"Company A's filing content"
DOC_B_CONTENT = b"Company B's filing content"


# --- Uploading a new document clears prior chat history and old retrieval state ---


def test_upload_with_empty_history_loads_immediately(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    path = _write_fixture(tmp_path, "10k.pdf", DOC_A_CONTENT)

    pending, confirm_row, store, doc_id, history, status, *_, doc_paths, last_sources = (
        app.handle_upload(path, [], {}, {})
    )

    assert pending is None
    assert confirm_row.visible is False
    assert isinstance(store, VectorStore)
    assert doc_id == _expected_doc_id(DOC_A_CONTENT)
    assert history == []  # fresh session, no carried-over chat
    assert doc_paths[doc_id] == path
    assert last_sources == []


def test_confirmed_upload_replaces_store_and_clears_chat(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    old_store = VectorStore()
    old_history = [
        {"role": "user", "content": "what is net debt?"},
        {"role": "assistant", "content": "$50B, see p.34"},
    ]
    new_path = _write_fixture(tmp_path, "new.pdf", DOC_B_CONTENT)

    _, confirm_row, _, _, _, _, *_ = app.handle_upload(new_path, old_history, {}, {})
    assert confirm_row.visible is True  # staged, not yet applied

    pending, confirm_row, store, doc_id, history, status, *_, doc_paths, last_sources = (
        app.confirm_upload(new_path, {}, {})
    )

    assert confirm_row.visible is False
    assert doc_id == _expected_doc_id(DOC_B_CONTENT)
    assert store is not old_store
    assert isinstance(store, VectorStore)
    assert history == []  # old conversation (with stale p.34 citation) is gone
    assert doc_paths[doc_id] == new_path
    assert last_sources == []  # stale citation buttons must not survive a document replace


# --- A cached metrics dashboard for a still-relevant document id survives a chat clear ---


A_DASHBOARD = ("<div>a's dashboard</div>", None, "✅ Extracted credit metrics for FY2025.")


def test_cached_metrics_survive_a_session_reset_to_another_document(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    doc_a_id = _expected_doc_id(DOC_A_CONTENT)
    metrics_cache = {doc_a_id: A_DASHBOARD}
    path_b = _write_fixture(tmp_path, "b.pdf", DOC_B_CONTENT)

    # Replacing the session with a *different* document must not touch a's entry.
    (
        _,
        _,
        _,
        doc_id,
        _,
        _,
        metrics_html,
        _,
        metrics_status,
        metrics_btn,
        doc_paths,
        last_sources,
    ) = app.confirm_upload(path_b, metrics_cache, {})

    assert doc_id == _expected_doc_id(DOC_B_CONTENT)
    assert doc_id != doc_a_id
    assert metrics_html == ""  # b.pdf has no cached dashboard yet
    assert metrics_status == "No metrics extracted yet."
    assert metrics_btn.interactive is True
    assert metrics_cache == {doc_a_id: A_DASHBOARD}
    assert doc_paths[doc_id] == path_b
    assert last_sources == []


def test_reuploading_a_cached_document_surfaces_its_dashboard(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    cached_html = A_DASHBOARD[0]
    doc_a_id = _expected_doc_id(DOC_A_CONTENT)
    metrics_cache = {doc_a_id: A_DASHBOARD}
    path_a = _write_fixture(tmp_path, "a.pdf", DOC_A_CONTENT)

    _, _, _, doc_id, _, _, metrics_html, _, metrics_status, metrics_btn, doc_paths, last_sources = (
        app.confirm_upload(path_a, metrics_cache, {})
    )

    assert doc_id == doc_a_id
    assert metrics_html == cached_html
    assert metrics_btn.interactive is False  # already extracted for this document
    assert doc_paths[doc_id] == path_a
    assert last_sources == []


def test_process_metrics_writes_cache_without_mutating_input_dict(monkeypatch, tmp_path):
    _mock_extraction(monkeypatch)
    original_cache = {}
    original_doc_paths = {}
    path = _write_fixture(tmp_path, "10k.pdf", DOC_A_CONTENT)

    html, chart_df, status, btn, new_cache, new_doc_paths = app.process_metrics(
        path, original_cache, original_doc_paths
    )

    expected_id = _expected_doc_id(DOC_A_CONTENT)
    assert original_cache == {}  # caller's dict is untouched
    assert original_doc_paths == {}  # caller's dict is untouched
    assert new_cache[expected_id] == (html, chart_df, status)
    assert new_doc_paths[expected_id] == path
    assert btn.interactive is False


def test_process_metrics_does_not_clobber_other_documents_cache_entry(monkeypatch, tmp_path):
    _mock_extraction(monkeypatch, fiscal_year=2025)
    old_dashboard = ("<div>old</div>", None, "✅ Extracted credit metrics for FY2024.")
    old_id = _expected_doc_id(b"some other, previously-uploaded filing")
    existing = {old_id: old_dashboard}
    old_doc_paths = {old_id: "/some/other/path.pdf"}
    path = _write_fixture(tmp_path, "10k.pdf", DOC_A_CONTENT)

    _, _, _, _, new_cache, new_doc_paths = app.process_metrics(path, existing, old_doc_paths)

    assert new_cache[old_id] == old_dashboard
    assert _expected_doc_id(DOC_A_CONTENT) in new_cache
    assert new_doc_paths[old_id] == "/some/other/path.pdf"  # not clobbered
    assert new_doc_paths[_expected_doc_id(DOC_A_CONTENT)] == path


# --- doc_id is a content hash: same content -> same id; same filename, different
# content -> different ids. This is the collision the metrics cache used to have
# when doc_id was just Path(path).name (see app._document_id's docstring). ---


def test_same_content_reuploaded_produces_the_same_doc_id(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    path = _write_fixture(tmp_path, "10k.pdf", DOC_A_CONTENT)

    _, _, _, doc_id_first, *_ = app.handle_upload(path, [], {}, {})
    _, _, _, doc_id_second, *_ = app.confirm_upload(path, {}, {})

    assert doc_id_first == doc_id_second == _expected_doc_id(DOC_A_CONTENT)


def test_different_content_same_filename_produces_different_doc_ids(monkeypatch, tmp_path):
    """The historical bug: two different uploads sharing a filename (e.g. two
    companies' filings both saved as "10-k.pdf") must not collide on doc_id --
    a collision would silently surface one document's cached metrics dashboard,
    or open the wrong document's page for a citation click, when the other is
    (re-)uploaded. doc_id is now a content hash, not the filename, so this
    can't happen."""
    _mock_build_index(monkeypatch)
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    path_a = _write_fixture(dir_a, "10-k.pdf", DOC_A_CONTENT)
    path_b = _write_fixture(dir_b, "10-k.pdf", DOC_B_CONTENT)

    _, _, _, doc_id_a, *_ = app.handle_upload(path_a, [], {}, {})
    _, _, _, doc_id_b, *_ = app.handle_upload(path_b, [], {}, {})

    assert doc_id_a != doc_id_b
    assert doc_id_a == _expected_doc_id(DOC_A_CONTENT)
    assert doc_id_b == _expected_doc_id(DOC_B_CONTENT)


# --- The confirmation path is triggered only when the existing conversation is non-empty ---


def test_no_confirmation_prompt_when_history_is_empty(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    path = _write_fixture(tmp_path, "10k.pdf", DOC_A_CONTENT)

    pending, confirm_row, *_ = app.handle_upload(path, [], {}, {})

    assert pending is None
    assert confirm_row.visible is False


def test_no_confirmation_prompt_when_history_is_none(monkeypatch, tmp_path):
    _mock_build_index(monkeypatch)
    path = _write_fixture(tmp_path, "10k.pdf", DOC_A_CONTENT)

    pending, confirm_row, *_ = app.handle_upload(path, None, {}, {})

    assert pending is None
    assert confirm_row.visible is False


def test_confirmation_prompt_shown_when_history_is_nonempty(monkeypatch):
    # This branch returns before _load_document/_document_id is ever called
    # (the file is only staged pending confirmation), so a bare string
    # stands in fine -- no file is opened.
    history = [{"role": "user", "content": "hi"}]

    pending, confirm_row, store, doc_id, chat_out, status, *_ = app.handle_upload(
        "new.pdf", history, {}, {}
    )

    assert pending == "new.pdf"
    assert confirm_row.visible is True
    # Nothing about the active session is touched yet -- these are "leave unchanged".
    assert store == gr.skip()
    assert doc_id == gr.skip()
    assert chat_out == gr.skip()


def test_cancel_upload_leaves_pending_and_session_untouched():
    pending, confirm_row, store, doc_id, chat_out, status, *_ = app.cancel_upload()

    assert pending is None
    assert confirm_row.visible is False
    assert store == gr.skip()
    assert doc_id == gr.skip()
    assert chat_out == gr.skip()
    assert "previous document" in status
