"""Session-state helpers for the single-document Q&A flow.

A "session" pairs an active document's retrieval state (a VectorStore) with
its conversation history. Uploading a document is either a *replace*
(single-document mode: start_new_session) or, in a future multi-document
mode, an *add* that would extend an existing session's documents while
keeping history intact. Keeping these as separate named operations now means
the add-source path can be introduced later without reworking upload
handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .vectorstore import VectorStore


@dataclass
class Session:
    """The active document and conversation state for single-document mode."""

    doc_id: str | None
    store: VectorStore | None
    history: list[dict] = field(default_factory=list)


def start_new_session(doc_id: str, store: VectorStore) -> Session:
    """Replace the active document, discarding prior conversation/retrieval state.

    This is the single-document "replace" path: the new document becomes the
    sole active document, and any previous chat history is dropped because it
    carries citations into a store that's no longer loaded. A future
    multi-document "add source" path would extend an existing Session's
    documents instead of replacing it, and would keep `history` intact --
    that path is not implemented here.
    """
    return Session(doc_id=doc_id, store=store, history=[])


def needs_confirmation(history: list[dict] | None) -> bool:
    """Whether replacing the active document should prompt before discarding history."""
    return bool(history)
