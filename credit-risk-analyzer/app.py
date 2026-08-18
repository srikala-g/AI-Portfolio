"""Gradio front end for the Credit Risk Analyzer app.

Run locally (recommended):  pip install -e .  then  python app.py
Deploy: push to a Hugging Face Space (SDK: gradio) with GEMINI_API_KEY as a secret.

The two lines below let `python app.py` work even without `pip install -e .`
(e.g. on Hugging Face Spaces, which installs requirements.txt but not the local
package). When the package IS installed, they are harmless.
"""

import hashlib
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

import gradio as gr  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from credit_risk_analyzer.metrics import (  # noqa: E402
    compute_all_derived_metrics,
    debt_maturity_chart_data,
    extract_line_items,
    render_dashboard_html,
)
from credit_risk_analyzer.pdf_view import (  # noqa: E402
    PageImageError,
    get_highlighted_page_image_path,
    get_page_image_path,
)
from credit_risk_analyzer.rag import answer_question, build_index  # noqa: E402
from credit_risk_analyzer.session import needs_confirmation, start_new_session  # noqa: E402

CONFIRM_MESSAGE = (
    "⚠️ Uploading a new document will start a new session and clear this "
    "conversation. Continue?"
)

# Delegated click listener for the credit-metrics dashboard's in-text
# citation links. This is a Python-declared, first-class parameter of the
# gr.HTML component constructor (js_on_load) -- separate from the
# dashboard's `value` HTML string that metrics.py builds -- not JavaScript
# smuggled into that sanitized content. Verified live: data-* attributes
# survive inside gr.HTML content, a single delegated listener bound at
# mount time keeps firing correctly across repeated `value` updates
# (re-extraction), and trigger()'s payload arrives in Python as evt.doc /
# evt.page via EventData.__getattr__. preventDefault() also stops the
# citation-link click from triggering the native <details>/<summary>
# toggle, since the citation line renders inside <summary> (see
# metrics.render_metric_tile_html).
#
# data-value is optional (metrics._render_citations_html only emits it when
# Citation.value is known); link.dataset.value is undefined for a link
# without it, and JSON.stringify drops undefined-valued keys entirely, so
# evt.value is simply absent (not None) on the Python side for those links
# -- on_metrics_citation_click reads it with getattr(..., None).
METRICS_JS_ON_LOAD = """
element.addEventListener('click', function(ev) {
    const link = ev.target.closest('.citation-link');
    if (!link) return;
    ev.preventDefault();
    trigger('click', {doc: link.dataset.doc, page: link.dataset.page, value: link.dataset.value});
});
"""


def _document_id(path: str) -> str:
    """Stable per-document id: a content hash, reused as the metrics-cache key
    and as the identity a citation button resolves to open the right PDF page.

    Hashing content (not the filename) matches the pattern
    embeddings._build_cache_key already uses for the embedding cache, and
    closes a real correctness gap: two different uploads that happen to
    share a filename (e.g. two companies' filings both saved as "10-k.pdf")
    previously collided on the same doc_id, silently surfacing one
    document's cached metrics dashboard -- and, now, would open the wrong
    document's page for a citation click -- when the other was re-uploaded.
    """
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _metrics_view(metrics_cache, doc_id):
    """Read-only lookup of a cached credit-metrics dashboard for doc_id.

    Never mutates metrics_cache -- a session reset must not evict entries.
    """
    cached = metrics_cache.get(doc_id)
    if cached is None:
        return "", None, "No metrics extracted yet.", gr.Button(interactive=True)
    html, chart_df, status = cached
    return html, chart_df, status, gr.Button(interactive=False)


def _load_document(path, metrics_cache, doc_paths, progress):
    """Index a PDF and start a fresh session for it (the single-document "replace" path).

    Returns the tuple of values for [store_state, doc_id_state, chatbot, status,
    metrics_html, maturity_chart, metrics_status, metrics_btn, doc_paths_state,
    last_sources_state]. Only reads metrics_cache to surface an existing
    dashboard for this doc_id, if any -- it never writes to or evicts from it.
    doc_paths is updated (non-mutating merge, same pattern as metrics_cache)
    with doc_id -> the real on-disk path, which is what a citation click
    later resolves through pdf_view.get_page_image_path to open the right
    page. last_sources resets to [] along with chat history: a "View p.N"
    button left over from the previous document's last answer would resolve
    against the *new* doc_id with a stale page number once clicked -- a
    plausible-looking wrong page, not a crash, so it must not survive a
    document replace.
    """
    progress(0, desc="Reading PDF…")
    try:

        def on_progress(done, total):
            progress(done / total, desc=f"Embedding chunks… {done}/{total}")

        store = build_index(path, on_progress=on_progress)
    except Exception as e:
        logger.exception("Failed to process uploaded PDF")
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            f"⚠️ Could not process this file: {e}",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    doc_id = _document_id(path)
    session = start_new_session(doc_id, store)
    doc_paths = {**doc_paths, doc_id: path}
    status_msg = f"✅ Indexed {len(store)} chunks — ready. Ask your question below."
    metrics_html, chart_df, metrics_status, metrics_btn = _metrics_view(
        metrics_cache, session.doc_id
    )
    return (
        session.store,
        session.doc_id,
        session.history,
        status_msg,
        metrics_html,
        chart_df,
        metrics_status,
        metrics_btn,
        doc_paths,
        [],  # last_sources: cleared, see docstring
    )


def handle_upload(file, history, metrics_cache, doc_paths, progress=gr.Progress()):  # noqa: B008
    """Route a new upload: load immediately, or stage it pending confirmation.

    A non-empty `history` means discarding it would silently lose a
    conversation, so the file is only staged here; confirm_upload/cancel_upload
    (wired to the confirm_row buttons) decide what happens to it.
    """
    if file is None:
        return (
            None,
            gr.Row(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Please upload a PDF to begin.",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    path = file if isinstance(file, str) else file.name

    if needs_confirmation(history):
        return (
            path,
            gr.Row(visible=True),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            "Confirm above to replace the current document.",
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )

    loaded = _load_document(path, metrics_cache, doc_paths, progress)
    return (None, gr.Row(visible=False), *loaded)


def confirm_upload(pending_path, metrics_cache, doc_paths, progress=gr.Progress()):  # noqa: B008
    """User confirmed the replace-and-clear prompt: load the staged document."""
    if not pending_path:
        return (
            None,
            gr.Row(visible=False),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),
        )
    loaded = _load_document(pending_path, metrics_cache, doc_paths, progress)
    return (None, gr.Row(visible=False), *loaded)


def cancel_upload():
    """User declined the prompt: the previous session is left exactly as it was."""
    return (
        None,
        gr.Row(visible=False),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        "Upload canceled — still using the previous document.",
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
        gr.skip(),
    )


def ask(store, question, history):
    """Answer a question against the indexed document.

    The third return value is the sorted list of source page numbers for
    this answer, fed to last_sources_state so the "View p.N" companion
    buttons below the chat (see the @gr.render block near the Ask Questions
    tab) reflect only the most recent answer -- historical messages in the
    chat stay non-interactive text, since gr.Chatbot can't host a real
    button inside an individual past bubble.
    """
    history = history or []
    if store is None:
        history.append({"role": "assistant", "content": "Upload and index a PDF first."})
        return history, "", []
    if not question or not question.strip():
        return history, "", gr.skip()

    try:
        result = answer_question(store, question)
    except Exception as exc:
        logger.exception("Failed to answer question")
        history.append({"role": "assistant", "content": f"⚠️ Could not answer that question: {exc}"})
        return history, "", []

    pages = sorted({c.page for c, _ in result.sources})
    sources_note = f"\n\n*Sources: page(s) {', '.join(map(str, pages))}*"

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result.text + sources_note})
    return history, "", pages


def process_metrics(file, metrics_cache, doc_paths, progress=gr.Progress()):  # noqa: B008
    """Extract raw line items via long-context Gemini, then compute ratios in Python.

    On success, caches the rendered dashboard under this document's id
    (derived straight from the uploaded file) so a later session reset --
    see session.start_new_session -- can't evict it: the dashboard is a
    per-document artifact, not conversation state.

    doc_paths is updated with doc_id -> path independently of the Q&A upload
    flow (handle_upload/_load_document does the same thing) so a citation in
    the metrics dashboard can resolve its source page even if this tab is
    used before the Q&A confirm-replace flow has completed for this file.

    The extract button stays enabled unless extraction actually succeeds -- so a
    missing file, an unparseable filing, or an API error all leave it clickable
    to retry, and only a successful extraction (or a cache hit on re-upload of
    the same document, via _metrics_view) disables it.
    """
    if file is None:
        return (
            "",
            None,
            "Please upload a PDF first.",
            gr.Button(interactive=True),
            metrics_cache,
            doc_paths,
        )
    progress(0, desc="Extracting line items from the filing…")
    try:
        path = file if isinstance(file, str) else file.name
        doc_id = _document_id(path)
        doc_paths = {**doc_paths, doc_id: path}

        def on_retry(attempt, max_retries, wait_seconds):
            progress(
                0,
                desc=(
                    f"Gemini rate limit hit — retrying in {wait_seconds:.0f}s "
                    f"(attempt {attempt + 1}/{max_retries})…"
                ),
            )

        financials = extract_line_items(path, on_retry=on_retry)
        if not financials.fiscal_years:
            return (
                "",
                None,
                "⚠️ Could not find fiscal-year figures in this filing.",
                gr.Button(interactive=True),
                metrics_cache,
                doc_paths,
            )

        progress(0.7, desc="Computing derived metrics…")
        fiscal_year = max(financials.fiscal_years)
        prior_years = sorted((y for y in financials.fiscal_years if y != fiscal_year), reverse=True)
        prior_fiscal_year = prior_years[0] if prior_years else None

        derived = compute_all_derived_metrics(financials, fiscal_year, prior_fiscal_year)
        html = render_dashboard_html(derived, fiscal_year, doc_id)
        chart_df = debt_maturity_chart_data(financials)
        status = f"✅ Extracted credit metrics for FY{fiscal_year}."
        metrics_cache = {**metrics_cache, doc_id: (html, chart_df, status)}
        return html, chart_df, status, gr.Button(interactive=False), metrics_cache, doc_paths
    except Exception as e:
        logger.exception("Failed to extract credit metrics")
        return (
            "",
            None,
            f"⚠️ Could not extract credit metrics: {e}",
            gr.Button(interactive=True),
            metrics_cache,
            doc_paths,
        )


def show_page(doc_id, page, doc_paths, value=None):
    """Rasterize and display the source PDF page a citation points to.

    Shared by both citation feeder paths -- the Q&A "View p.N" buttons
    (page arrives as a real int, from gr.State; never pass a value -- Q&A
    citations are retrieved chunks, not a single as-reported figure) and
    the credit-metrics dashboard's in-text citation links (page arrives as
    a string, read off a data-page DOM attribute by app.py's js_on_load
    click listener; value likewise, when the citation carries one) -- see
    pdf_view.py for the rendering + disk-cache logic. Returns updates for
    (page_viewer_image, page_viewer_caption, page_viewer_group-visibility).

    When `value` is given, this attempts a best-effort highlight box around
    that figure on the page (pdf_view.get_highlighted_page_image_path) --
    strictly additive: on any failure to find a confident single match, it
    returns None and this function falls straight through to the same
    plain get_page_image_path call it would have made without a value.
    Worst case is identical to not having passed a value at all.
    """
    if not doc_id or not page:
        return None, "", gr.Group(visible=False)

    try:
        page = int(page)
    except (TypeError, ValueError):
        return None, f"⚠️ Could not open this citation: invalid page ({page!r}).", gr.Group(
            visible=True
        )

    path = doc_paths.get(doc_id)
    if not path:
        return (
            None,
            "⚠️ Could not find the source document for this citation "
            "(it may be from a different session).",
            gr.Group(visible=True),
        )

    image_path = None
    if value is not None:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            value_float = None
        if value_float is not None:
            image_path = get_highlighted_page_image_path(path, doc_id, page, value_float)

    if image_path is None:
        try:
            image_path = get_page_image_path(path, doc_id, page)
        except PageImageError as exc:
            return None, f"⚠️ Could not open page {page}: {exc}", gr.Group(visible=True)

    caption = f"**{pathlib.Path(path).name}** — page {page}"
    return image_path, caption, gr.Group(visible=True)


def on_metrics_citation_click(doc_paths, evt: gr.EventData):
    """Adapter for the metrics dashboard's in-text citation links.

    evt.doc/evt.page arrive via the dashboard's js_on_load delegated click
    listener (see METRICS_JS_ON_LOAD below) as raw strings read off a
    clicked <a class="citation-link" data-doc data-page [data-value]>
    element's DOM attributes -- reuses show_page's own coercion/error/
    highlight-fallback handling rather than duplicating it here. evt.value
    is absent (not None) when the link had no data-value attribute (see
    METRICS_JS_ON_LOAD's comment), hence getattr with a None default.
    """
    return show_page(evt.doc, evt.page, doc_paths, value=getattr(evt, "value", None))


# Restyles the Q&A "View p.N" companion controls (real gr.Buttons, wired to
# show_page the same as before) from bordered buttons to inline citation
# links -- the gr.Chatbot fallback's visual counterpart to the metrics
# dashboard's true in-text <a class="citation-link"> links, since Chatbot
# has no mechanism to host a real link inside a message bubble itself.
CITATION_LINK_BUTTON_CSS = """
.citation-link-btn {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
    color: inherit !important;
    text-decoration: underline;
    padding: 0 !important;
    min-width: unset !important;
    font-weight: 400 !important;
}
.citation-link-btn:hover { opacity: 0.7; }
/* Gradio's default Row gives every child flex-grow:1/flex-basis:0%, so the
   label and each p.N link get stretched into equal-width columns spanning
   the full row -- a small `gap` alone doesn't fix that, since the visible
   spread comes from each box's own width, not the gap between boxes.
   `flex: none` on every direct child makes each one only as wide as its
   content, so the tightened gap is what actually determines the spacing,
   and the whole cluster reads as one inline group instead of a scattered
   button bar. */
.citation-row { gap: 0.4rem !important; align-items: center !important; }
.citation-row > * { flex: 0 1 auto !important; width: auto !important; }
.citation-row .block { padding: 0 !important; margin: 0 !important; }
"""

with gr.Blocks(title="Credit Risk Analyzer (RAG)") as demo:
    gr.Markdown(
        "# Credit Risk Analyzer\n"
        "Upload a financial document (10-K, prospectus, earnings transcript) and "
        "ask questions. Answers are grounded in the document with page citations."
    )
    store_state = gr.State(None)
    doc_id_state = gr.State(None)
    pending_path_state = gr.State(None)
    metrics_cache_state = gr.State({})
    # doc_id (content hash) -> the real on-disk path Gradio stored the upload
    # at. This is what a citation click resolves through to rasterize a page
    # (see show_page/pdf_view.get_page_image_path). Updated the same
    # non-mutating way as metrics_cache_state -- one document's entry never
    # evicts another's.
    doc_paths_state = gr.State({})
    # Page numbers cited by the most recent Q&A answer, feeding the "View
    # p.N" button row below the chat input.
    last_sources_state = gr.State([])

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        status = gr.Markdown("No document loaded yet.")

    with gr.Row(visible=False) as confirm_row:
        gr.Markdown(CONFIRM_MESSAGE)
        confirm_btn = gr.Button("Continue", variant="primary")
        cancel_btn = gr.Button("Cancel")

    with gr.Tabs() as tabs:
        with gr.Tab("Ask Questions"):
            chatbot = gr.Chatbot(label="Conversation", height=420)
            question_box = gr.Textbox(
                label="Your question",
                placeholder="e.g. What are the key risk factors? What was net revenue?",
            )
            ask_btn = gr.Button("Ask", variant="primary")

            @gr.render(inputs=[last_sources_state])
            def _render_source_buttons(pages):
                """Clickable page numbers for the latest answer only -- a
                gr.Chatbot bubble can't host a real link (gr.Chatbot has no
                js_on_load/server_functions hook the way gr.HTML does, and
                its .select() event only reports which whole message was
                clicked, not which link inside it -- confirmed empirically,
                not assumed), so this is a companion row below the chat,
                restyled to read as inline citation links rather than a
                button row. Reruns whenever last_sources_state changes (i.e.
                after every answer); each link's page number is fixed at
                render time via gr.State(page), while doc_id/doc_paths are
                read live at click time so this always targets the current
                document.
                """
                if not pages:
                    return
                with gr.Row(elem_classes=["citation-row"]):
                    gr.Markdown("**View source page:**")
                    for p in pages:
                        page_btn = gr.Button(f"p.{p}", size="sm", elem_classes=["citation-link-btn"])
                        page_btn.click(
                            show_page,
                            inputs=[doc_id_state, gr.State(p), doc_paths_state],
                            outputs=[page_viewer_image, page_viewer_caption, page_viewer_group],
                        )

        with gr.Tab("Credit Metrics"):
            gr.Markdown(
                "Extracts raw reported line items directly from the filing with a "
                "long-context model call (not the Q&A retrieval path above), then "
                "computes every ratio in Python so figures are never LLM arithmetic."
            )
            metrics_btn = gr.Button("Extract Credit Metrics", variant="primary")
            metrics_status = gr.Markdown("No metrics extracted yet.")
            # js_on_load wires the in-text citation-link click handling for
            # this component -- see METRICS_JS_ON_LOAD and
            # on_metrics_citation_click. metrics.py's dashboard HTML (the
            # `value` this component displays) never contains a <script>
            # tag; the click mechanism is declared here, at the component
            # level, not injected into that sanitized string.
            metrics_html = gr.HTML(js_on_load=METRICS_JS_ON_LOAD)
            maturity_chart = gr.BarPlot(
                x="year",
                y="principal",
                title="Debt Maturity Wall (principal due by year, $M)",
                x_title="Year",
                y_title="Principal ($M)",
            )

    # Shared page-image viewer -- both the Q&A "View p.N" links and the
    # credit-metrics dashboard's in-text citation links target this same
    # pair of outputs via show_page, regardless of which tab is active.
    # Starts hidden; a citation click reveals it.
    with gr.Group(visible=False) as page_viewer_group:
        gr.Markdown("### Source page")
        page_viewer_caption = gr.Markdown()
        page_viewer_image = gr.Image(label="Source page", interactive=False, show_label=False)

    metrics_html.click(
        on_metrics_citation_click,
        inputs=[doc_paths_state],
        outputs=[page_viewer_image, page_viewer_caption, page_viewer_group],
    )

    # A page opened from one tab's citations reads as out of place once the
    # user has switched away from that tab -- hide it on every tab switch;
    # the next citation click reopens it in the right context.
    tabs.select(lambda: gr.Group(visible=False), outputs=[page_viewer_group])

    upload_outputs = [
        pending_path_state,
        confirm_row,
        store_state,
        doc_id_state,
        chatbot,
        status,
        metrics_html,
        maturity_chart,
        metrics_status,
        metrics_btn,
        doc_paths_state,
        last_sources_state,
    ]
    pdf_input.upload(
        handle_upload,
        inputs=[pdf_input, chatbot, metrics_cache_state, doc_paths_state],
        outputs=upload_outputs,
    )
    confirm_btn.click(
        confirm_upload,
        inputs=[pending_path_state, metrics_cache_state, doc_paths_state],
        outputs=upload_outputs,
    )
    cancel_btn.click(cancel_upload, outputs=upload_outputs)

    ask_outputs = [chatbot, question_box, last_sources_state]
    ask_btn.click(ask, inputs=[store_state, question_box, chatbot], outputs=ask_outputs)
    question_box.submit(ask, inputs=[store_state, question_box, chatbot], outputs=ask_outputs)
    metrics_btn.click(
        process_metrics,
        inputs=[pdf_input, metrics_cache_state, doc_paths_state],
        outputs=[
            metrics_html,
            maturity_chart,
            metrics_status,
            metrics_btn,
            metrics_cache_state,
            doc_paths_state,
        ],
        show_progress_on=metrics_status,
    )


if __name__ == "__main__":
    demo.queue().launch(css=CITATION_LINK_BUTTON_CSS)
