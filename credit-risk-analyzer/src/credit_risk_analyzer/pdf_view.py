"""Rasterize a single PDF page for the citation page-viewer.

A citation only ever carries a page number, and both the Q&A path (page tags
on chunks) and the credit-metrics path (page tags on line items) target the
same original PDF. This module turns (path to that PDF, 1-based page number)
into a viewable PNG, with a disk cache keyed by (doc_id, page) so clicking
the same citation twice doesn't re-rasterize -- the same spirit as the
embedding cache in embeddings.py and the extraction cache in metrics.py.

Rendering uses pypdfium2 (a pure-pip wheel bundling PDFium) rather than
shelling out to poppler's pdftoppm: no system package to provision on the
deploy target (Hugging Face Spaces), which is exactly the kind of thing that
works locally and silently breaks in a container.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw

from .config import PAGE_IMAGE_CACHE_DIR, PAGE_IMAGE_DPI

logger = logging.getLogger(__name__)


class PageImageError(RuntimeError):
    """A citation's (doc_id, page) could not be resolved to a page image."""


def _cache_path(doc_id: str, page: int) -> Path:
    return Path(PAGE_IMAGE_CACHE_DIR) / f"{doc_id}_{page}.png"


def get_page_image_path(pdf_path: str, doc_id: str, page: int) -> str:
    """Return a filesystem path to a rasterized PNG of `page` (1-based) from
    the PDF at `pdf_path`. `doc_id` is the caller's content-hash identity for
    that PDF (see app._document_id) -- used only to key the cache, not to
    look anything up itself.

    Raises PageImageError if the file can't be opened or the page number is
    out of range, rather than letting a pdfium exception (or an off-by-one
    IndexError) surface directly to the UI layer.
    """
    cache_file = _cache_path(doc_id, page)
    if cache_file.exists():
        return str(cache_file)

    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except (FileNotFoundError, OSError) as exc:
        raise PageImageError(f"Could not open the source PDF: {exc}") from exc

    try:
        if page < 1 or page > len(pdf):
            raise PageImageError(
                f"Page {page} is out of range for this document (it has {len(pdf)} pages)."
            )
        rendered = pdf[page - 1].render(scale=PAGE_IMAGE_DPI / 72)
        image = rendered.to_pil()
    except pdfium.PdfiumError as exc:
        raise PageImageError(f"Could not render page {page}: {exc}") from exc
    finally:
        pdf.close()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(cache_file)
    return str(cache_file)


# ---------------------------------------------------------------------------
# Best-effort value highlighting (Route A: search the page's own text for
# the value's printed form, box it if -- and only if -- there's exactly one
# match). This is a convenience layer on top of the page-level citation
# above, which remains the real guarantee: every function below fails
# silently (returns None), never raising, so a highlighting miss can never
# break opening the plain page. See app.show_page for how the fallback is
# wired: it always has the plain get_page_image_path result to fall back to.
# ---------------------------------------------------------------------------

_HIGHLIGHT_COLOR = (220, 38, 38)  # a readable red, distinct from the dashboard's palette
_HIGHLIGHT_WIDTH = 3
_HIGHLIGHT_PAD = 3


def _highlight_cache_path(doc_id: str, page: int, value: float) -> Path:
    # Keyed by (doc_id, page, value) per the design brief -- hash the value
    # into the filename rather than embedding it raw, since a raw float repr
    # isn't a safe/stable filename component.
    value_key = hashlib.sha256(f"{value:.6f}".encode()).hexdigest()[:16]
    return Path(PAGE_IMAGE_CACHE_DIR) / f"{doc_id}_{page}_hl_{value_key}.png"


def _value_search_variants(value: float) -> list[str]:
    """Plausible as-printed forms of a raw LineItem.value.

    `,.0f` (not the naive `,`) is required: LineItem.value is a float, and
    `f"{value:,}"` on 12350.0 produces "12,350.0", which never matches the
    printed "12,350" -- confirmed against real filing text before this was
    built. The parenthesized form covers accounting-style negatives (e.g.
    "(12,715)"); not needed by any case in the Apple fixture, but kept as
    insurance for filings that do print outflows that way.
    """
    plain = f"{abs(value):,.0f}"
    return [plain, f"({plain})"]


def _find_value_box(
    pdf: pdfium.PdfDocument, page_index: int, value: float
) -> tuple[float, float, float, float] | None:
    """Search page `page_index` (0-based) for `value`'s printed form and
    return a single pixel-space box (left, top, right, bottom) -- but only
    when exactly one match exists for the first variant that matches at
    all. Zero matches (the value isn't printed verbatim -- e.g. a filing
    describing a figure in rounded prose like "$2.6 billion" instead of a
    table figure) or multiple matches (the same digits appearing more than
    once on the page, which real filings do -- e.g. a balance sheet's
    "Total assets" figure recurring on the "Total liabilities and
    shareholders' equity" line by the accounting identity) both return
    None: this deliberately doesn't try to disambiguate multiple matches
    (e.g. by proximity to a row label), since the only label text available
    is the citation's source *statement* name, which is identical for both
    matches in exactly the real case that motivated this rule -- picking
    one anyway would be presenting an unverified guess as a source, which
    is worse than no highlight.
    """
    page = pdf[page_index]
    textpage = page.get_textpage()
    page_h = page.get_height()
    scale = PAGE_IMAGE_DPI / 72

    for variant in _value_search_variants(value):
        searcher = textpage.search(variant)
        matches = []
        m = searcher.get_next()
        while m:
            matches.append(m)
            m = searcher.get_next()

        if not matches:
            continue  # try the next format variant
        if len(matches) > 1:
            logger.info(
                "Value %r matched %d times on page %d -- ambiguous, no highlight",
                variant,
                len(matches),
                page_index + 1,
            )
            return None

        charindex, count = matches[0]
        n_rects = textpage.count_rects(charindex, count)
        if n_rects < 1:
            return None

        # PDFium does not reliably merge a match into one rect -- e.g. a
        # 7-character match can come back as 7 per-character rects even on
        # a single line with no wrap (confirmed against a real match: the
        # same "111,482" string that count_rects(...)==1 for one page's
        # font/spacing came back as 7 for another). This is still exactly
        # one confirmed match (checked above), just possibly described by
        # several adjacent rects -- union them into the one bounding box
        # that covers the whole match, rather than treating the rect count
        # as a second ambiguity signal.
        rects = [textpage.get_rect(i) for i in range(n_rects)]
        left = min(r[0] for r in rects)
        bottom = min(r[1] for r in rects)
        right = max(r[2] for r in rects)
        top = max(r[3] for r in rects)
        return (
            left * scale,
            (page_h - top) * scale,
            right * scale,
            (page_h - bottom) * scale,
        )

    return None  # no variant matched at all


def get_highlighted_page_image_path(
    pdf_path: str, doc_id: str, page: int, value: float
) -> str | None:
    """Best-effort: the same rasterized page as get_page_image_path, with a
    box drawn around a confident single match for `value`'s printed form.

    Returns None -- never raises -- on any failure: the file can't be
    opened, the page is out of range, or no confident single match is
    found. The caller (app.show_page) always has get_page_image_path's
    plain result to fall back to, so a problem here can only ever result in
    the page opening without a box, exactly like it did before this
    feature existed.
    """
    cache_file = _highlight_cache_path(doc_id, page, value)
    if cache_file.exists():
        return str(cache_file)

    try:
        base_image_path = get_page_image_path(pdf_path, doc_id, page)
    except PageImageError:
        return None

    try:
        pdf = pdfium.PdfDocument(pdf_path)
    except (FileNotFoundError, OSError):
        return None

    try:
        if page < 1 or page > len(pdf):
            return None
        box = _find_value_box(pdf, page - 1, value)
    except pdfium.PdfiumError:
        logger.info("Value search failed for page %d, doc %s", page, doc_id, exc_info=True)
        box = None
    finally:
        pdf.close()

    if box is None:
        return None

    image = Image.open(base_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = box
    draw.rectangle(
        [left - _HIGHLIGHT_PAD, top - _HIGHLIGHT_PAD, right + _HIGHLIGHT_PAD, bottom + _HIGHLIGHT_PAD],
        outline=_HIGHLIGHT_COLOR,
        width=_HIGHLIGHT_WIDTH,
    )

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(cache_file)
    return str(cache_file)
