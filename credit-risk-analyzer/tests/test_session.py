"""Tests for the pure session-reset helpers (no Gradio, no network)."""

from credit_risk_analyzer.session import Session, needs_confirmation, start_new_session
from credit_risk_analyzer.vectorstore import VectorStore


def test_start_new_session_returns_fresh_empty_history():
    store = VectorStore()
    session = start_new_session("10k.pdf", store)

    assert isinstance(session, Session)
    assert session.doc_id == "10k.pdf"
    assert session.store is store
    assert session.history == []


def test_start_new_session_does_not_reuse_prior_history():
    # start_new_session takes no "prior session" argument at all -- there is
    # no code path for it to carry old history forward.
    store_a = VectorStore()
    session_a = start_new_session("a.pdf", store_a)
    session_a.history.append({"role": "user", "content": "what is net debt?"})

    store_b = VectorStore()
    session_b = start_new_session("b.pdf", store_b)

    assert session_b.history == []
    assert session_a.history != session_b.history


def test_needs_confirmation_false_for_empty_or_missing_history():
    assert needs_confirmation([]) is False
    assert needs_confirmation(None) is False


def test_needs_confirmation_true_for_nonempty_history():
    history = [{"role": "user", "content": "hi"}]
    assert needs_confirmation(history) is True
