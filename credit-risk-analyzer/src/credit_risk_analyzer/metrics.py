"""Credit metrics: long-context line-item extraction + Python-computed ratios.

This is a deliberate second path alongside the RAG Q&A pipeline (rag.py), not a
reuse of it. Metrics need every relevant number in front of the model at once
(revenue, debt, cash flow, the maturity schedule) rather than the top-k
similarity-retrieved snippets RAG hands back, and large filings would blow
through the free-tier embedding rate limit if indexed just to pull ~30 numbers.
So the whole parsed filing (page-tagged, like rag._format_context) goes to
Gemini in one call.

The LLM's job stops at reading off raw, as-reported figures with their page
number. Every ratio, sum, and derived figure below is computed in plain Python
from those line items, never by the model -- in a credit tool a wrong number is
worse than none, and LLMs are unreliable at arithmetic.

Null propagation is strict: if a line item a derived metric depends on is
missing, that metric -- and anything computed from it -- resolves to `None`
with a typed `unavailable_reason` plus a human-readable `reason` explaining
what's missing, never a value that silently substitutes zero for an unreported
figure. Two reasons are distinguished (see `UnavailableReason`):
INPUT_MISSING (a required line item isn't disclosed in this filing) and
NOT_APPLICABLE (the metric doesn't structurally apply to this filing, or the
computation is undefined for the reported values, e.g. a zero denominator).
This keeps the metric catalog generic across filings that report differently
-- a metric that doesn't apply to a given filing is distinguished from one
whose data simply wasn't found.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from google import genai
from google.genai import types

from .config import METRICS_CACHE_DIR, METRICS_MODEL, METRICS_SYSTEM_PROMPT, METRICS_TEMPERATURE
from .ingest import load_pdf_pages
from .retry import call_with_retry

# Raw line items extracted verbatim from the filing (never computed by the LLM).
# `interest_expense_estimate` is not one of the user-specified raw metrics --
# it's added to support the netted-interest fallback described below.
LINE_ITEM_KEYS = [
    "total_revenue",
    "products_revenue",
    "services_revenue",
    "cost_of_sales",
    "gross_margin",
    "operating_income",
    "net_income",
    "research_and_development",
    "selling_general_administrative",
    "depreciation_amortization",
    "share_based_comp",
    "interest_expense",
    "interest_expense_estimate",
    "cash_and_equivalents",
    "current_marketable_securities",
    "noncurrent_marketable_securities",
    "total_current_assets",
    "total_current_liabilities",
    "inventories",
    "accounts_receivable",
    "accounts_payable",
    "current_term_debt",
    "noncurrent_term_debt",
    "commercial_paper",
    "total_assets",
    "total_liabilities",
    "total_shareholders_equity",
    "operating_cash_flow",
    "capex",
    "dividends_paid",
    "share_repurchases",
]

_SCHEMA_EXAMPLE = """{
  "fiscal_years": [2025, 2024],
  "line_items": {
    "total_revenue": [
      {"fiscal_year": 2025, "value": 416161, "unit": "USD_MILLIONS",
       "source_statement": "Consolidated Statements of Operations", "page": 33},
      {"fiscal_year": 2024, "value": 391035, "unit": "USD_MILLIONS",
       "source_statement": "Consolidated Statements of Operations", "page": 33}
    ],
    "interest_expense": [
      {"fiscal_year": 2025, "value": null, "unit": null, "source_statement": null, "page": null}
    ]
  },
  "debt_maturity_schedule": [
    {"year": 2026, "value": 12393, "unit": "USD_MILLIONS",
     "source_statement": "Notes to Financial Statements - Term Debt", "page": 50}
  ]
}"""


@dataclass
class LineItem:
    """A single raw, as-reported figure and where it came from."""

    value: float | None
    unit: str | None
    fiscal_year: int
    source_statement: str | None
    page: int | None


@dataclass
class MaturityYear:
    """One row of the debt maturity schedule (a future calendar year -> principal due)."""

    year: int
    value: float | None
    unit: str | None
    source_statement: str | None
    page: int | None


@dataclass
class ExtractedFinancials:
    """Everything pulled from a filing before any ratio is computed."""

    fiscal_years: list[int]
    line_items: dict[str, dict[int, LineItem]]  # name -> fiscal_year -> LineItem
    debt_maturity_schedule: list[MaturityYear]


@dataclass
class Citation:
    statement: str | None
    page: int | None
    # The as-reported figure this citation traces to, when known (i.e. it
    # came from a LineItem, not synthesized). Metadata only -- used by the
    # dashboard's best-effort value-highlighting (see pdf_view.py /
    # app.show_page), never by any formula or availability check.
    value: float | None = None


class UnavailableReason(Enum):
    """Why a derived metric has no value -- kept distinct so a caller (or the
    UI) never has to guess whether a gap is a data problem or an expected
    structural absence.

    INPUT_MISSING: a required raw line item or upstream derived metric wasn't
    disclosed in this filing (e.g. interest expense not separately reported).
    NOT_APPLICABLE: the metric doesn't structurally apply to this filing (e.g.
    a product/services revenue mix when the filing reports no such split), or
    the computation is undefined for the reported values (e.g. a zero
    denominator).
    """

    INPUT_MISSING = "input_missing"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class InputTrace:
    """One named input exactly as _combine (or compute_coverage's manual
    branches) consumed it -- for the explanation bubble's calculation view.
    Captured at computation time, never re-derived: `label` is the same
    variable name that appears in DerivedMetric.formula, `value`/`unit`
    mirror exactly what was passed to (or missing from) the formula
    function, and `citation` is the input's own page-level citation when it
    came from a raw LineItem. A nested DerivedMetric input (the one-level
    cascade case, e.g. net_debt as an input to net_debt_to_ebitda) has no
    citation -- it isn't a single-page figure, it's itself a calculation
    the reader can drill into via that metric's own tile.
    """

    label: str
    value: float | None
    unit: str | None
    citation: Citation | None = None
    # Set only when value is None: the same per-input reason text
    # _combine's own missing-detection loop already produces for the
    # joined DerivedMetric.reason string, just kept structured per input.
    unavailable_reason: str | None = None


@dataclass
class DerivedMetric:
    """A ratio/sum computed in Python, with the formula and provenance that produced it."""

    name: str
    value: float | None
    unit: str | None
    formula: str
    fiscal_year: int
    citations: list[Citation] = field(default_factory=list)
    reason: str | None = None  # human-readable explanation, if value is None
    unavailable_reason: UnavailableReason | None = None  # typed category, if value is None
    estimated: bool = False  # value relies on a labeled approximation, e.g. netted interest expense
    # Every named input as _combine actually used it, in formula order --
    # empty by default so any DerivedMetric built without tracing (e.g.
    # _not_applicable's structural skip, which never had inputs to gather)
    # degrades to "no calculation shown" in the UI, not a crash.
    inputs: list[InputTrace] = field(default_factory=list)


_client = None


def _get_client() -> genai.Client:
    """Create the client on first use so importing this module needs no API key."""
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def build_document_text(pages: list[tuple[int, str]]) -> str:
    """Page-tagged full filing text, so the model can report the page it read a figure from."""
    return "\n\n".join(f"[p. {page}]\n{text}" for page, text in pages)


def build_extraction_prompt(document_text: str) -> str:
    return (
        "Extract these raw line items for every fiscal year present in the filing "
        f"below: {', '.join(LINE_ITEM_KEYS)}.\n\n"
        "`interest_expense` is often netted inside a line such as 'Other income/"
        "(expense), net' rather than disclosed separately -- if you cannot find a "
        "clean interest expense figure, set it to null. In that case also look in "
        "the term-debt note for the interest payable within the next 12 months "
        "(a memo disclosure near the debt maturity schedule, distinct from debt "
        "principal) and report it as `interest_expense_estimate`.\n\n"
        "Also extract the future debt maturity schedule (each future calendar "
        "year mapped to the principal due) from the term-debt note.\n\n"
        "Return STRICT JSON only -- no markdown code fences, no commentary -- "
        "matching exactly this shape:\n\n"
        f"{_SCHEMA_EXAMPLE}\n\n"
        f"Filing text (each page tagged with its page number):\n\n{document_text}"
    )


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def _coerce_float(value: object) -> float | None:
    return None if value is None else float(value)


def _coerce_int(value: object) -> int | None:
    return None if value is None else int(value)


def parse_llm_response(raw_text: str) -> ExtractedFinancials:
    """Defensively parse the model's JSON: strip fences, tolerate missing keys."""
    cleaned = _strip_code_fences(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {exc}") from exc

    fiscal_years = [int(y) for y in data.get("fiscal_years", [])]

    raw_line_items = data.get("line_items", {})
    line_items: dict[str, dict[int, LineItem]] = {}
    for key in LINE_ITEM_KEYS:
        by_year: dict[int, LineItem] = {}
        for entry in raw_line_items.get(key, []) or []:
            fy = int(entry["fiscal_year"])
            by_year[fy] = LineItem(
                value=_coerce_float(entry.get("value")),
                unit=entry.get("unit"),
                fiscal_year=fy,
                source_statement=entry.get("source_statement"),
                page=_coerce_int(entry.get("page")),
            )
        line_items[key] = by_year

    schedule = [
        MaturityYear(
            year=int(entry["year"]),
            value=_coerce_float(entry.get("value")),
            unit=entry.get("unit"),
            source_statement=entry.get("source_statement"),
            page=_coerce_int(entry.get("page")),
        )
        for entry in data.get("debt_maturity_schedule", []) or []
    ]

    return ExtractedFinancials(
        fiscal_years=fiscal_years, line_items=line_items, debt_maturity_schedule=schedule
    )


def _extraction_cache_key(prompt: str) -> str:
    """Hash the exact prompt (+ model/temperature) so any change to the filing,
    the prompt template, or the model automatically invalidates old cache
    entries rather than silently serving a stale extraction."""
    payload = f"{METRICS_MODEL}|{METRICS_TEMPERATURE}|{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extraction_cache_path(key: str) -> Path:
    return Path(METRICS_CACHE_DIR) / f"{key}.json"


def _load_cached_extraction(key: str) -> str | None:
    path = _extraction_cache_path(key)
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _save_cached_extraction(key: str, raw_text: str) -> None:
    path = _extraction_cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw_text, encoding="utf-8")


def extract_line_items(pdf_path: str, on_retry=None) -> ExtractedFinancials:
    """Ingest a filing and extract raw line items via a single long-context Gemini call.

    The raw model response is cached to disk keyed by the exact prompt sent
    (see ``_extraction_cache_key``), so re-analyzing the same filing -- e.g.
    rehearsing a demo -- is instant and makes zero API calls instead of
    resending the whole filing and risking the free-tier rate limit again.

    ``on_retry(attempt, max_retries, wait_seconds)`` is forwarded to
    ``call_with_retry`` so a caller with a UI can show that a rate-limit/server
    retry is in progress instead of looking stuck for the wait duration.
    """
    pages = load_pdf_pages(pdf_path)
    if not pages:
        raise ValueError("No extractable text found. Is this a scanned PDF? (OCR needed.)")

    prompt = build_extraction_prompt(build_document_text(pages))
    cache_key = _extraction_cache_key(prompt)

    cached_text = _load_cached_extraction(cache_key)
    if cached_text is not None:
        return parse_llm_response(cached_text)

    resp = call_with_retry(
        lambda: _get_client().models.generate_content(
            model=METRICS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=METRICS_SYSTEM_PROMPT,
                temperature=METRICS_TEMPERATURE,
            ),
        ),
        on_retry=on_retry,
    )
    financials = parse_llm_response(resp.text)  # parse first: never cache invalid JSON
    _save_cached_extraction(cache_key, resp.text)
    return financials


# ---------------------------------------------------------------------------
# Derived metrics -- computed in Python, never by the LLM
# ---------------------------------------------------------------------------

LineItems = dict[str, dict[int, LineItem]]
_Input = LineItem | DerivedMetric | None


def _li(items: LineItems, name: str, fiscal_year: int) -> LineItem | None:
    return items.get(name, {}).get(fiscal_year)


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[tuple[str | None, int | None]] = set()
    out = []
    for c in citations:
        key = (c.statement, c.page)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _gather_citations(inputs: dict[str, _Input]) -> list[Citation]:
    citations: list[Citation] = []
    for v in inputs.values():
        if isinstance(v, LineItem) and v.value is not None:
            citations.append(Citation(v.source_statement, v.page, v.value))
        elif isinstance(v, DerivedMetric):
            citations.extend(v.citations)
    return _dedupe_citations(citations)


def _combine(
    name: str,
    formula: str,
    fiscal_year: int,
    inputs: dict[str, _Input],
    fn,
    unit: str | None,
    estimated: bool = False,
) -> DerivedMetric:
    """Compute a derived metric, or return None + a typed reason if it can't be.

    Never substitutes a missing input with zero -- if any required raw line item
    or upstream derived metric is unavailable, this metric (and anything built
    on top of it) becomes unavailable too, tagged INPUT_MISSING with the
    specific input(s) named. A zero denominator is tagged NOT_APPLICABLE
    instead: the inputs are present, but the ratio is undefined for those
    values, which is a different situation from data simply not being found.

    Also builds `traces` (one InputTrace per named input, in formula order)
    in this same pass -- retaining, not re-deriving, exactly what this loop
    already decided about each input, so the explanation bubble can later
    show the calculation without recomputing anything.
    """
    missing: list[str] = []
    traces: list[InputTrace] = []
    for key, v in inputs.items():
        if v is None:
            missing.append(f"{key} (not extracted from filing)")
            traces.append(
                InputTrace(label=key, value=None, unit=None, unavailable_reason="not extracted from filing")
            )
        elif v.value is None:
            reason = v.reason if isinstance(v, DerivedMetric) and v.reason else "not found in filing"
            missing.append(f"{key} ({reason})")
            traces.append(InputTrace(label=key, value=None, unit=v.unit, unavailable_reason=reason))
        elif isinstance(v, LineItem):
            traces.append(
                InputTrace(
                    label=key,
                    value=v.value,
                    unit=v.unit,
                    citation=Citation(v.source_statement, v.page, v.value),
                )
            )
        else:  # an available DerivedMetric input -- one-level cascade, no citation of its own
            traces.append(InputTrace(label=key, value=v.value, unit=v.unit))

    if missing:
        return DerivedMetric(
            name=name,
            value=None,
            unit=unit,
            formula=formula,
            fiscal_year=fiscal_year,
            citations=[],
            reason="not available: " + "; ".join(missing),
            unavailable_reason=UnavailableReason.INPUT_MISSING,
            inputs=traces,
        )

    values = {k: v.value for k, v in inputs.items()}
    try:
        result = fn(values)
    except ZeroDivisionError:
        return DerivedMetric(
            name=name,
            value=None,
            unit=unit,
            formula=formula,
            fiscal_year=fiscal_year,
            citations=_gather_citations(inputs),
            reason="not applicable: division by zero",
            unavailable_reason=UnavailableReason.NOT_APPLICABLE,
            inputs=traces,
        )

    return DerivedMetric(
        name=name,
        value=result,
        unit=unit,
        formula=formula,
        fiscal_year=fiscal_year,
        citations=_gather_citations(inputs),
        estimated=estimated,
        inputs=traces,
    )


def _not_applicable(
    name: str, formula: str, fiscal_year: int, unit: str | None, reason: str
) -> DerivedMetric:
    """A metric that structurally doesn't apply to this filing -- distinct from
    a metric whose required input just wasn't found (see `UnavailableReason`)."""
    return DerivedMetric(
        name=name,
        value=None,
        unit=unit,
        formula=formula,
        fiscal_year=fiscal_year,
        citations=[],
        reason=reason,
        unavailable_reason=UnavailableReason.NOT_APPLICABLE,
    )


def _has_revenue_breakdown(items: LineItems) -> bool:
    """Whether this filing discloses any product/services revenue split at all,
    in any fiscal year -- distinct from one year's figure simply being missing
    despite the filing generally reporting that kind of breakdown. Filings that
    don't segment revenue this way (most filings) never satisfy this, which is
    what makes the services-mix metrics below NOT_APPLICABLE rather than a
    perpetual INPUT_MISSING."""
    return any(
        li.value is not None
        for key in ("products_revenue", "services_revenue")
        for li in items.get(key, {}).values()
    )


_NO_REVENUE_BREAKDOWN_REASON = (
    "not applicable: this filing does not report a product/services (or "
    "similar) revenue breakdown"
)


def compute_profitability(
    items: LineItems, fiscal_year: int, prior_fiscal_year: int | None = None
) -> dict[str, DerivedMetric]:
    revenue = _li(items, "total_revenue", fiscal_year)
    services_revenue = _li(items, "services_revenue", fiscal_year)
    gross_margin = _li(items, "gross_margin", fiscal_year)
    operating_income = _li(items, "operating_income", fiscal_year)
    net_income = _li(items, "net_income", fiscal_year)
    # Whether this filing discloses a revenue-type split at all -- if not, the
    # services-mix metrics below are structurally not applicable rather than
    # perpetually "missing" (see `_has_revenue_breakdown`).
    has_revenue_breakdown = _has_revenue_breakdown(items)

    services_mix_formula = "services_revenue / total_revenue * 100"
    out = {
        "gross_margin_pct": _combine(
            "gross_margin_pct",
            "gross_margin / total_revenue * 100",
            fiscal_year,
            {"gross_margin": gross_margin, "total_revenue": revenue},
            lambda v: v["gross_margin"] / v["total_revenue"] * 100,
            "%",
        ),
        "operating_margin_pct": _combine(
            "operating_margin_pct",
            "operating_income / total_revenue * 100",
            fiscal_year,
            {"operating_income": operating_income, "total_revenue": revenue},
            lambda v: v["operating_income"] / v["total_revenue"] * 100,
            "%",
        ),
        "net_margin_pct": _combine(
            "net_margin_pct",
            "net_income / total_revenue * 100",
            fiscal_year,
            {"net_income": net_income, "total_revenue": revenue},
            lambda v: v["net_income"] / v["total_revenue"] * 100,
            "%",
        ),
        "services_mix_pct": (
            _combine(
                "services_mix_pct",
                services_mix_formula,
                fiscal_year,
                {"services_revenue": services_revenue, "total_revenue": revenue},
                lambda v: v["services_revenue"] / v["total_revenue"] * 100,
                "%",
            )
            if has_revenue_breakdown
            else _not_applicable(
                "services_mix_pct",
                services_mix_formula,
                fiscal_year,
                "%",
                _NO_REVENUE_BREAKDOWN_REASON,
            )
        ),
    }

    if prior_fiscal_year is not None:
        prior_revenue = _li(items, "total_revenue", prior_fiscal_year)
        prior_services_revenue = _li(items, "services_revenue", prior_fiscal_year)
        out["revenue_yoy_growth_pct"] = _combine(
            "revenue_yoy_growth_pct",
            f"(total_revenue[{fiscal_year}] - total_revenue[{prior_fiscal_year}])"
            f" / total_revenue[{prior_fiscal_year}] * 100",
            fiscal_year,
            {"revenue": revenue, "prior_revenue": prior_revenue},
            lambda v: (v["revenue"] - v["prior_revenue"]) / v["prior_revenue"] * 100,
            "%",
        )
        services_growth_formula = (
            f"(services_revenue[{fiscal_year}] - services_revenue[{prior_fiscal_year}])"
            f" / services_revenue[{prior_fiscal_year}] * 100"
        )
        out["services_revenue_growth_pct"] = (
            _combine(
                "services_revenue_growth_pct",
                services_growth_formula,
                fiscal_year,
                {
                    "services_revenue": services_revenue,
                    "prior_services_revenue": prior_services_revenue,
                },
                lambda v: (v["services_revenue"] - v["prior_services_revenue"])
                / v["prior_services_revenue"]
                * 100,
                "%",
            )
            if has_revenue_breakdown
            else _not_applicable(
                "services_revenue_growth_pct",
                services_growth_formula,
                fiscal_year,
                "%",
                _NO_REVENUE_BREAKDOWN_REASON,
            )
        )

    return out


def compute_leverage(items: LineItems, fiscal_year: int) -> dict[str, DerivedMetric]:
    current_term_debt = _li(items, "current_term_debt", fiscal_year)
    noncurrent_term_debt = _li(items, "noncurrent_term_debt", fiscal_year)
    commercial_paper = _li(items, "commercial_paper", fiscal_year)
    cash = _li(items, "cash_and_equivalents", fiscal_year)
    current_securities = _li(items, "current_marketable_securities", fiscal_year)
    noncurrent_securities = _li(items, "noncurrent_marketable_securities", fiscal_year)
    operating_income = _li(items, "operating_income", fiscal_year)
    da = _li(items, "depreciation_amortization", fiscal_year)

    total_debt = _combine(
        "total_debt",
        "current_term_debt + noncurrent_term_debt + commercial_paper",
        fiscal_year,
        {
            "current_term_debt": current_term_debt,
            "noncurrent_term_debt": noncurrent_term_debt,
            "commercial_paper": commercial_paper,
        },
        lambda v: v["current_term_debt"] + v["noncurrent_term_debt"] + v["commercial_paper"],
        "USD_MILLIONS",
    )
    cash_and_securities = _combine(
        "cash_and_securities",
        "cash_and_equivalents + current_marketable_securities + noncurrent_marketable_securities",
        fiscal_year,
        {
            "cash_and_equivalents": cash,
            "current_marketable_securities": current_securities,
            "noncurrent_marketable_securities": noncurrent_securities,
        },
        lambda v: v["cash_and_equivalents"]
        + v["current_marketable_securities"]
        + v["noncurrent_marketable_securities"],
        "USD_MILLIONS",
    )
    # Negative net_debt means net cash -- rendering decides how to label the sign;
    # this stays a plain signed dollar figure so the UI can show "Net cash: $X".
    net_debt = _combine(
        "net_debt",
        "total_debt - cash_and_securities",
        fiscal_year,
        {"total_debt": total_debt, "cash_and_securities": cash_and_securities},
        lambda v: v["total_debt"] - v["cash_and_securities"],
        "USD_MILLIONS",
    )
    ebitda = _combine(
        "ebitda",
        "operating_income + depreciation_amortization",
        fiscal_year,
        {"operating_income": operating_income, "depreciation_amortization": da},
        lambda v: v["operating_income"] + v["depreciation_amortization"],
        "USD_MILLIONS",
    )
    gross_debt_to_ebitda = _combine(
        "gross_debt_to_ebitda",
        "total_debt / ebitda",
        fiscal_year,
        {"total_debt": total_debt, "ebitda": ebitda},
        lambda v: v["total_debt"] / v["ebitda"],
        "x",
    )
    # When net_debt < 0 this is a net-cash position; the ratio itself is still
    # computed (it's arithmetically valid), but it is not a meaningful leverage
    # multiple -- rendering must show "net cash", not a negative "x" multiple.
    net_debt_to_ebitda = _combine(
        "net_debt_to_ebitda",
        "net_debt / ebitda",
        fiscal_year,
        {"net_debt": net_debt, "ebitda": ebitda},
        lambda v: v["net_debt"] / v["ebitda"],
        "x",
    )

    return {
        "total_debt": total_debt,
        "cash_and_securities": cash_and_securities,
        "net_debt": net_debt,
        "ebitda": ebitda,
        "gross_debt_to_ebitda": gross_debt_to_ebitda,
        "net_debt_to_ebitda": net_debt_to_ebitda,
    }


def compute_coverage(
    items: LineItems, fiscal_year: int, ebitda: DerivedMetric
) -> dict[str, DerivedMetric]:
    """EBITDA / interest expense, falling back to a labeled estimate if interest
    expense is netted inside a combined line (e.g. "Other income/(expense),
    net") rather than disclosed separately."""
    interest_expense = _li(items, "interest_expense", fiscal_year)
    interest_estimate = _li(items, "interest_expense_estimate", fiscal_year)

    estimated = False
    if interest_expense is not None and interest_expense.value is not None:
        chosen: LineItem | None = interest_expense
        formula = "ebitda / interest_expense"
    elif interest_estimate is not None and interest_estimate.value is not None:
        chosen = interest_estimate
        estimated = True
        formula = (
            "ebitda / interest_expense_estimate (estimate: interest expense is netted "
            "elsewhere and not separately disclosed; using interest payable within 12 "
            "months from the term-debt note as an approximation)"
        )
    else:
        chosen = None
        formula = "ebitda / interest_expense"

    # This function builds DerivedMetric directly rather than through
    # _combine (its "try actual, fall back to estimate" branching doesn't
    # fit _combine's generic all-inputs-required shape) -- so each branch
    # traces its own inputs by hand, from values already in scope here
    # (ebitda.value/unit/reason, chosen.value/unit/source_statement/page),
    # the same information _combine's own tracing would capture.
    interest_label = "interest_expense_estimate" if estimated else "interest_expense"

    if ebitda.value is None:
        metric = DerivedMetric(
            name="ebitda_to_interest_expense",
            value=None,
            unit="x",
            formula=formula,
            fiscal_year=fiscal_year,
            citations=[],
            reason=f"not available: ebitda ({ebitda.reason})",
            unavailable_reason=UnavailableReason.INPUT_MISSING,
            inputs=[InputTrace(label="ebitda", value=None, unit=ebitda.unit, unavailable_reason=ebitda.reason)],
        )
    elif chosen is None:
        interest_missing_reason = (
            "not separately disclosed (possibly netted inside a combined line such as "
            "'Other income/(expense), net') and no debt-note interest-payable estimate "
            "was found either"
        )
        metric = DerivedMetric(
            name="ebitda_to_interest_expense",
            value=None,
            unit="x",
            formula=formula,
            fiscal_year=fiscal_year,
            citations=list(ebitda.citations),
            reason=f"not available: interest expense is {interest_missing_reason}",
            unavailable_reason=UnavailableReason.INPUT_MISSING,
            inputs=[
                InputTrace(label="ebitda", value=ebitda.value, unit=ebitda.unit),
                InputTrace(
                    label="interest_expense",
                    value=None,
                    unit=None,
                    unavailable_reason=interest_missing_reason,
                ),
            ],
        )
    elif chosen.value == 0:
        metric = DerivedMetric(
            name="ebitda_to_interest_expense",
            value=None,
            unit="x",
            formula=formula,
            fiscal_year=fiscal_year,
            citations=list(ebitda.citations),
            reason=(
                "not applicable: interest expense is reported as 0, so the "
                "coverage ratio is undefined"
            ),
            unavailable_reason=UnavailableReason.NOT_APPLICABLE,
            inputs=[
                InputTrace(label="ebitda", value=ebitda.value, unit=ebitda.unit),
                InputTrace(
                    label=interest_label,
                    value=chosen.value,
                    unit=chosen.unit,
                    citation=Citation(chosen.source_statement, chosen.page, chosen.value),
                ),
            ],
        )
    else:
        citations = _dedupe_citations(
            [*ebitda.citations, Citation(chosen.source_statement, chosen.page, chosen.value)]
        )
        reason = (
            "estimate: interest expense is netted elsewhere and not separately "
            "disclosed; this uses interest payable within 12 months from the "
            "term-debt note, not the company's actual cash interest expense"
            if estimated
            else None
        )
        metric = DerivedMetric(
            name="ebitda_to_interest_expense",
            value=ebitda.value / chosen.value,
            unit="x",
            formula=formula,
            fiscal_year=fiscal_year,
            citations=citations,
            reason=reason,
            estimated=estimated,
            inputs=[
                InputTrace(label="ebitda", value=ebitda.value, unit=ebitda.unit),
                InputTrace(
                    label=interest_label,
                    value=chosen.value,
                    unit=chosen.unit,
                    citation=Citation(chosen.source_statement, chosen.page, chosen.value),
                ),
            ],
        )

    return {"ebitda_to_interest_expense": metric}


def compute_liquidity(items: LineItems, fiscal_year: int) -> dict[str, DerivedMetric]:
    total_current_assets = _li(items, "total_current_assets", fiscal_year)
    total_current_liabilities = _li(items, "total_current_liabilities", fiscal_year)
    inventories = _li(items, "inventories", fiscal_year)
    accounts_receivable = _li(items, "accounts_receivable", fiscal_year)
    accounts_payable = _li(items, "accounts_payable", fiscal_year)
    revenue = _li(items, "total_revenue", fiscal_year)
    cost_of_sales = _li(items, "cost_of_sales", fiscal_year)

    # Below-1.0x is not automatically flagged as distress -- a negative cash
    # conversion cycle (fast customer collections, slow supplier payments) can
    # make a sub-1.0x current ratio benign rather than a liquidity warning.
    # No hardcoded threshold here; the UI surfaces context instead of a verdict.
    current_ratio = _combine(
        "current_ratio",
        "total_current_assets / total_current_liabilities",
        fiscal_year,
        {
            "total_current_assets": total_current_assets,
            "total_current_liabilities": total_current_liabilities,
        },
        lambda v: v["total_current_assets"] / v["total_current_liabilities"],
        "x",
    )
    quick_ratio = _combine(
        "quick_ratio",
        "(total_current_assets - inventories) / total_current_liabilities",
        fiscal_year,
        {
            "total_current_assets": total_current_assets,
            "inventories": inventories,
            "total_current_liabilities": total_current_liabilities,
        },
        lambda v: (v["total_current_assets"] - v["inventories"])
        / v["total_current_liabilities"],
        "x",
    )
    # DSO/DIO/DPO use year-end balances (not average of beginning/ending) --
    # a standard simplification given a single filing snapshot.
    dso = _combine(
        "days_sales_outstanding",
        "accounts_receivable / total_revenue * 365",
        fiscal_year,
        {"accounts_receivable": accounts_receivable, "total_revenue": revenue},
        lambda v: v["accounts_receivable"] / v["total_revenue"] * 365,
        "days",
    )
    dio = _combine(
        "days_inventory_outstanding",
        "inventories / cost_of_sales * 365",
        fiscal_year,
        {"inventories": inventories, "cost_of_sales": cost_of_sales},
        lambda v: v["inventories"] / v["cost_of_sales"] * 365,
        "days",
    )
    dpo = _combine(
        "days_payable_outstanding",
        "accounts_payable / cost_of_sales * 365",
        fiscal_year,
        {"accounts_payable": accounts_payable, "cost_of_sales": cost_of_sales},
        lambda v: v["accounts_payable"] / v["cost_of_sales"] * 365,
        "days",
    )
    cash_conversion_cycle = _combine(
        "cash_conversion_cycle",
        "days_sales_outstanding + days_inventory_outstanding - days_payable_outstanding",
        fiscal_year,
        {
            "days_sales_outstanding": dso,
            "days_inventory_outstanding": dio,
            "days_payable_outstanding": dpo,
        },
        lambda v: v["days_sales_outstanding"]
        + v["days_inventory_outstanding"]
        - v["days_payable_outstanding"],
        "days",
    )

    return {
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "days_sales_outstanding": dso,
        "days_inventory_outstanding": dio,
        "days_payable_outstanding": dpo,
        "cash_conversion_cycle": cash_conversion_cycle,
    }


def compute_cash_flow(
    items: LineItems, fiscal_year: int, total_debt: DerivedMetric
) -> dict[str, DerivedMetric]:
    operating_cash_flow = _li(items, "operating_cash_flow", fiscal_year)
    capex = _li(items, "capex", fiscal_year)
    net_income = _li(items, "net_income", fiscal_year)
    da = _li(items, "depreciation_amortization", fiscal_year)
    sbc = _li(items, "share_based_comp", fiscal_year)

    free_cash_flow = _combine(
        "free_cash_flow",
        "operating_cash_flow - capex",
        fiscal_year,
        {"operating_cash_flow": operating_cash_flow, "capex": capex},
        lambda v: v["operating_cash_flow"] - v["capex"],
        "USD_MILLIONS",
    )
    ffo = _combine(
        "ffo",
        "net_income + depreciation_amortization + share_based_comp",
        fiscal_year,
        {"net_income": net_income, "depreciation_amortization": da, "share_based_comp": sbc},
        lambda v: v["net_income"] + v["depreciation_amortization"] + v["share_based_comp"],
        "USD_MILLIONS",
    )
    ffo_to_debt_pct = _combine(
        "ffo_to_debt_pct",
        "ffo / total_debt * 100",
        fiscal_year,
        {"ffo": ffo, "total_debt": total_debt},
        lambda v: v["ffo"] / v["total_debt"] * 100,
        "%",
    )
    fcf_to_debt_pct = _combine(
        "fcf_to_debt_pct",
        "free_cash_flow / total_debt * 100",
        fiscal_year,
        {"free_cash_flow": free_cash_flow, "total_debt": total_debt},
        lambda v: v["free_cash_flow"] / v["total_debt"] * 100,
        "%",
    )

    return {
        "free_cash_flow": free_cash_flow,
        "ffo": ffo,
        "ffo_to_debt_pct": ffo_to_debt_pct,
        "fcf_to_debt_pct": fcf_to_debt_pct,
    }


def compute_capital_return(
    items: LineItems, fiscal_year: int, free_cash_flow: DerivedMetric
) -> dict[str, DerivedMetric]:
    dividends_paid = _li(items, "dividends_paid", fiscal_year)
    share_repurchases = _li(items, "share_repurchases", fiscal_year)

    capital_return_total = _combine(
        "capital_return_total",
        "dividends_paid + share_repurchases",
        fiscal_year,
        {"dividends_paid": dividends_paid, "share_repurchases": share_repurchases},
        lambda v: v["dividends_paid"] + v["share_repurchases"],
        "USD_MILLIONS",
    )
    capital_return_vs_fcf_pct = _combine(
        "capital_return_vs_fcf_pct",
        "capital_return_total / free_cash_flow * 100",
        fiscal_year,
        {"capital_return_total": capital_return_total, "free_cash_flow": free_cash_flow},
        lambda v: v["capital_return_total"] / v["free_cash_flow"] * 100,
        "%",
    )

    return {
        "capital_return_total": capital_return_total,
        "capital_return_vs_fcf_pct": capital_return_vs_fcf_pct,
    }


def compute_all_derived_metrics(
    financials: ExtractedFinancials, fiscal_year: int, prior_fiscal_year: int | None = None
) -> dict[str, dict[str, DerivedMetric]]:
    """Compute every grouped derived metric for one fiscal year."""
    items = financials.line_items
    leverage = compute_leverage(items, fiscal_year)
    cash_flow = compute_cash_flow(items, fiscal_year, leverage["total_debt"])
    return {
        "leverage": leverage,
        "coverage": compute_coverage(items, fiscal_year, leverage["ebitda"]),
        "profitability": compute_profitability(items, fiscal_year, prior_fiscal_year),
        "liquidity": compute_liquidity(items, fiscal_year),
        "cash_flow": cash_flow,
        "capital_return": compute_capital_return(
            items, fiscal_year, cash_flow["free_cash_flow"]
        ),
    }


def debt_maturity_chart_data(financials: ExtractedFinancials):
    """Year -> principal rows for the maturity-wall bar chart, dropping unavailable years."""
    import pandas as pd

    rows = [
        {"year": str(m.year), "principal": m.value}
        for m in sorted(financials.debt_maturity_schedule, key=lambda m: m.year)
        if m.value is not None
    ]
    return pd.DataFrame(rows, columns=["year", "principal"])


# ---------------------------------------------------------------------------
# HTML rendering for the Gradio dashboard tiles
# ---------------------------------------------------------------------------

GROUP_TITLES = {
    "leverage": "Leverage & Net Debt",
    "coverage": "Interest Coverage",
    "profitability": "Profitability",
    "liquidity": "Liquidity",
    "cash_flow": "Cash Flow",
    "capital_return": "Capital Return",
}
GROUP_ORDER = ["leverage", "coverage", "profitability", "liquidity", "cash_flow", "capital_return"]

_DASHBOARD_STYLE = """
<style>
.metrics-dashboard { font-family: inherit; }
.metrics-dashboard h2 { margin-bottom: 0.5rem; }
.metrics-dashboard h3 { margin: 1.2rem 0 0.5rem; font-size: 1.05rem; }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 0.75rem; }
.metric-tile { border: 1px solid rgba(128,128,128,0.35); border-radius: 8px;
  padding: 0.6rem 0.8rem; }
.metric-tile > summary { cursor: pointer; }
.metric-tile > summary::-webkit-details-marker { color: rgba(128,128,128,0.6); }
.metric-title { font-size: 0.78rem; opacity: 0.75; margin-bottom: 0.2rem;
  display: inline-flex; align-items: center; gap: 0.3rem; }
.metric-info { font-size: 0.78rem; opacity: 0.55; }
/* A real computed value is bold, full-size, full-contrast (default text
   color -- no blue: it hurt readability). "Not available"/"Not applicable"
   (.metric-na below) is deliberately smaller, lighter-weight, and faded --
   that three-way distinction (size + weight + opacity) is what carries the
   calculated-vs-unavailable contrast now, not color. */
.metric-value { font-size: 1.3rem; font-weight: 600; }
.metric-value.metric-na { font-size: 1rem; font-weight: 500; opacity: 0.6; }
/* Estimated values (e.g. the netted-interest-expense coverage fallback)
   keep their distinct goldenrod -- the same shade as the "Estimated" badge
   below -- so the number itself still signals the caveat, not only the
   badge text. !important is required: the Gradio HTML component nests this
   markup inside a `.prose` wrapper, and Gradio's own stylesheet ships a
   `.gradio-container-<ver> .prose *` rule (two classes) that outranks a
   plain one-class selector and silently repaints it back to the default
   body text color. Verified against a live Gradio render, not just a
   standalone HTML file. */
.metric-value-estimated { color: #b8860b !important; }
.metric-citation { font-size: 0.72rem; opacity: 0.6; margin-top: 0.3rem; }
.metric-reason { font-size: 0.72rem; opacity: 0.6; margin-top: 0.2rem; font-style: italic; }
.metric-estimate { font-size: 0.72rem; color: #b8860b; margin-top: 0.2rem; }
.metric-explanation { font-size: 0.75rem; opacity: 0.85; margin-top: 0.5rem;
  padding-top: 0.4rem; border-top: 1px solid rgba(128,128,128,0.25); }
.metric-explanation p { margin: 0 0 0.3rem; }
.metric-explanation-label { opacity: 0.7; }
.metric-formula, .metric-explanation-citation { margin-top: 0.2rem; }
.metric-formula code { font-size: 0.7rem; }
/* The calculation view: formula (above) + one line per traced input +
   the result. Kept visually compact (small text, tight spacing) since a
   metric can have several inputs -- density is the layout answer to
   noise here, not hiding inputs. */
.metric-calc-inputs { margin-top: 0.35rem; }
.metric-calc-input { font-size: 0.72rem; margin-top: 0.15rem; }
.metric-calc-input-missing { opacity: 0.7; }
.metric-calc-missing { font-style: italic; }
.metric-calc-result { font-size: 0.72rem; font-weight: 600; margin-top: 0.35rem; }
/* Each citation is its own clickable link (see _render_citations_html) --
   opens the source page in the shared viewer via app.py's delegated
   js_on_load listener on this dashboard's gr.HTML component. Matches the
   Q&A companion links' treatment (CITATION_LINK_BUTTON_CSS in app.py):
   default text color + underline, not link-blue -- still reads as
   clickable via the underline, without the harsh default <a> blue.
   !important is required for the same reason as .metric-value-estimated
   above: this markup sits inside Gradio's `.prose` wrapper, and Gradio's
   own stylesheet ships `.gradio-container-<ver> .prose a { color:
   var(--link-text-color) }` -- two classes plus an element selector, which
   outranks a plain one-class `.citation-link` selector and repaints it
   back to Gradio's default link blue. Verified against a live Gradio
   render (inspected matched CSS rules on a rendered <a class="citation-
   link">), not assumed. */
.citation-link { color: inherit !important; text-decoration: underline; cursor: pointer; }
.citation-link:hover { opacity: 0.7; }
</style>
"""

# Per-metric explanation copy for the on-demand disclosure in each tile.
# Centralized here (same spirit as the prompts in config.py) rather than
# scattered through the render template. Seeded from the metric tables in
# README.md; keyed by DerivedMetric.name.
METRIC_EXPLANATIONS: dict[str, str] = {
    # Profitability
    "gross_margin_pct": "Revenue left after cost of goods sold, as a % of revenue.",
    "operating_margin_pct": (
        "Profit after operating expenses, before interest and tax, as a % of revenue."
    ),
    "net_margin_pct": "Bottom-line profit as a % of revenue.",
    "services_mix_pct": (
        "Share of revenue coming from services vs. products, where the filing reports "
        "that split. Not applicable for filings without a product/services (or similar) "
        "revenue breakdown."
    ),
    "revenue_yoy_growth_pct": "Revenue growth vs. the prior fiscal year.",
    "services_revenue_growth_pct": (
        "Services-revenue growth vs. the prior fiscal year, where that split is "
        "reported. Not applicable otherwise."
    ),
    # Leverage & net debt
    "total_debt": (
        "All interest-bearing borrowings: current + noncurrent term debt + "
        "commercial paper."
    ),
    "cash_and_securities": (
        "Liquid assets available to offset debt: cash + current + noncurrent "
        "marketable securities."
    ),
    "net_debt": "Total debt minus cash & securities; negative means a net cash position.",
    "ebitda": (
        "Operating income + depreciation & amortization — a proxy for operating "
        "cash-generating capacity."
    ),
    "gross_debt_to_ebitda": (
        "Years of EBITDA needed to repay all debt — a core leverage multiple "
        "lenders watch."
    ),
    "net_debt_to_ebitda": (
        'Same, netting out cash; a net-cash position is labeled "N/M (net cash)", '
        "not shown as a negative multiple."
    ),
    # Interest coverage
    "ebitda_to_interest_expense": (
        "How many times over operating earnings cover interest obligations; low "
        "values signal debt-service stress. Falls back to a labeled estimate "
        "(interest payable within 12 months, from the debt note) when interest "
        "expense is netted elsewhere in the filing rather than disclosed separately."
    ),
    # Liquidity
    "current_ratio": "Current assets / current liabilities — ability to cover near-term obligations.",
    "quick_ratio": "Same, excluding inventory (the least liquid current asset).",
    "days_sales_outstanding": "Average days to collect receivables.",
    "days_inventory_outstanding": "Average days inventory sits before being sold.",
    "days_payable_outstanding": "Average days taken to pay suppliers.",
    "cash_conversion_cycle": (
        "DSO + DIO − DPO: net days cash is tied up in operations. Can be negative "
        "— a sign of favorable supplier/customer terms, not distress."
    ),
    # Cash flow
    "free_cash_flow": (
        "Operating cash flow minus capex — cash available for debt service and "
        "shareholder returns."
    ),
    "ffo": (
        "Net income + D&A + share-based comp — a cash-oriented earnings measure "
        "common in credit analysis."
    ),
    "ffo_to_debt_pct": "A standard rating-agency cash-flow-coverage-of-leverage metric.",
    "fcf_to_debt_pct": (
        "Free cash flow relative to total debt — roughly how fast debt could be "
        "repaid from discretionary cash flow."
    ),
    # Capital return
    "capital_return_total": (
        "Dividends paid + share repurchases — total cash returned to shareholders."
    ),
    "capital_return_vs_fcf_pct": (
        "Share of free cash flow being returned to shareholders vs. retained or "
        "used for debt paydown."
    ),
}


def format_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "Not available"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "days":
        return f"{value:.0f} days"
    if unit == "USD_MILLIONS":
        if abs(value) >= 1000:
            return f"${value / 1000:,.1f}B"
        return f"${value:,.0f}M"
    return f"{value:,.2f}"


def format_input_value(value: float | None, unit: str | None) -> str:
    """Calculation-bubble input-line formatting -- unlike format_value,
    USD_MILLIONS is never billion-converted, regardless of magnitude. This
    is the reconciliation fix: a page-highlight boxes the as-printed figure
    (e.g. "95,281"), so an input line has to show that same plain-millions
    form ("$95,281M"), not the tile's billion-scaled convenience format
    ("$95.3B") -- otherwise a correct highlight looks like it doesn't match
    what's on screen. %, x, and days are unaffected (format_value never
    magnitude-scales those), so they're delegated straight through.

    Used ONLY for input lines. The result line always calls format_value
    directly (see _metric_result_text) -- never this function -- so it
    stays byte-identical to the tile's own value text.
    """
    if value is None:
        return "—"
    if unit == "USD_MILLIONS":
        return f"${value:,.0f}M"
    return format_value(value, unit)


def format_citation(citation: Citation) -> str:
    if citation.statement is None and citation.page is None:
        return ""
    statement = citation.statement or "Source"
    return f"{statement}, p.{citation.page}" if citation.page is not None else statement


def _citation_link_html(citation: Citation, doc_id: str, text: str) -> str:
    """A single clickable citation link with the given display text.

    Shared by the tile's "Source: ..." line (_render_citations_html, text =
    the citation's own label) and the calculation bubble's input lines
    (text = the input's formatted value) so both use the exact same
    data-doc/data-page/data-value attribute shape app.py's js_on_load
    listener expects -- one link-building implementation, not two that
    could drift apart.
    """
    if citation.page is None:
        return text
    value_attr = f' data-value="{citation.value}"' if citation.value is not None else ""
    return f'<a href="#" class="citation-link" data-doc="{doc_id}" data-page="{citation.page}"{value_attr}>{text}</a>'


def _render_citations_html(citations: list[Citation], doc_id: str) -> str:
    """Each citation as its own clickable page link, joined by "; ".

    A citation carrying a page number becomes an <a class="citation-link"
    data-doc="..." data-page="...">; app.py wires a single delegated
    js_on_load click listener on the dashboard's gr.HTML component that
    reads these two data attributes and opens the page in the shared
    viewer (see show_page). A citation with no page number (format_citation
    still returns a label, e.g. just the statement name) renders as plain
    text, since there's nothing to open. Each citation on a tile is its own
    link -- not one link around a joined string -- because two citations on
    the same metric can point at different pages (e.g. one line item from
    the income statement, another from a cash-flow note).

    When the citation also carries a value (Citation.value -- the
    as-reported figure it traces to), the link additionally carries
    data-value, so app.py's click handler can attempt a best-effort
    highlight box around that figure on the opened page (see
    pdf_view.get_highlighted_page_image_path). A citation with no
    resolvable value (e.g. deduped down to one whose value didn't survive,
    or a citation built without one) omits data-value entirely -- the page
    still opens, just without an attempted highlight.
    """
    parts = []
    for c in citations:
        label = format_citation(c)
        if not label:
            continue
        parts.append(_citation_link_html(c, doc_id, label))
    return "; ".join(parts)


def _metric_result_text(metric: DerivedMetric) -> str:
    """The exact text used for this metric's value, wherever it's shown --
    the tile's headline value AND the calculation bubble's result line.
    Reusing this one function is what guarantees the two can never disagree
    (e.g. via different rounding): there is exactly one place that turns a
    DerivedMetric into display text, and both callers go through it.
    """
    if metric.value is None:
        return (
            "Not applicable"
            if metric.unavailable_reason == UnavailableReason.NOT_APPLICABLE
            else "Not available"
        )
    # Negative net_debt is a net-cash position, not negative leverage --
    # shown as "Net cash" instead of a nonsensical negative multiple.
    if metric.name == "net_debt" and metric.value < 0:
        return f"Net cash: {format_value(-metric.value, metric.unit)}"
    if metric.name == "net_debt_to_ebitda" and metric.value < 0:
        return "N/M (net cash position)"
    return format_value(metric.value, metric.unit)


def _render_input_trace_html(trace: InputTrace, doc_id: str) -> str:
    """One input line in the calculation bubble: `label = value`.

    A found input's value is formatted in plain millions (format_input_value
    -- never billion-converted, see its docstring) and, when it traces to a
    raw LineItem's page, is itself the clickable citation link (reusing
    _citation_link_html, the same data-doc/data-page/data-value mechanism
    the tile's own citations use, so it highlights on click identically). A
    missing input shows its own typed reason instead of a value -- this is
    what makes an unavailable metric's cascade legible: which inputs were
    found and which weren't, side by side. A nested DerivedMetric input
    (one-level cascade) shows its value with no link, since it isn't a
    single-page figure.
    """
    if trace.value is None:
        reason = trace.unavailable_reason or "not available"
        return (
            f'<div class="metric-calc-input metric-calc-input-missing">'
            f'{trace.label} = <span class="metric-calc-missing">{reason}</span></div>'
        )
    value_text = format_input_value(trace.value, trace.unit)
    if trace.citation is not None:
        value_text = _citation_link_html(trace.citation, doc_id, value_text)
    return f'<div class="metric-calc-input">{trace.label} = {value_text}</div>'


def _render_calculation_html(metric: DerivedMetric, doc_id: str) -> str:
    """Formula, each traced input, and the result -- the metric's actual
    calculation, assembled verbatim from what _combine (or compute_
    coverage's manual branches) already captured on this DerivedMetric.
    Never recomputes or independently reformats the arithmetic: the result
    line calls _metric_result_text, the exact function the tile's headline
    value uses, so the two can't diverge.

    Reads only fields already present on the DerivedMetric -- formula,
    inputs, value -- so this stays a pure render-layer helper with no reach
    into the compute layer itself. `inputs` is empty for a DerivedMetric
    that never had inputs to trace (e.g. _not_applicable's structural
    skip), in which case only the formula line shows.
    """
    parts = []
    if metric.formula:
        parts.append(
            f'<div class="metric-formula">'
            f'<span class="metric-explanation-label">Formula:</span> '
            f"<code>{metric.formula}</code></div>"
        )
    if metric.inputs:
        input_rows = "".join(_render_input_trace_html(t, doc_id) for t in metric.inputs)
        parts.append(f'<div class="metric-calc-inputs">{input_rows}</div>')
        parts.append(
            f'<div class="metric-calc-result">'
            f'<span class="metric-explanation-label">Result:</span> {_metric_result_text(metric)}</div>'
        )
    return "".join(parts)


def render_metric_tile_html(metric: DerivedMetric, doc_id: str) -> str:
    title = metric.name.replace("_", " ").title()

    if metric.value is None:
        na_label = _metric_result_text(metric)
        # Left in its existing muted styling -- smaller, lighter-weight, and
        # faded relative to the calculated-value branch below, which is what
        # now carries the calculated-vs-unavailable contrast (see
        # .metric-value / .metric-value.metric-na in _DASHBOARD_STYLE).
        body = f'<div class="metric-value metric-na">{na_label}</div>'
        reason_text = metric.reason or "missing required inputs"
        detail = f'<div class="metric-reason">{reason_text}</div>'
    else:
        # _metric_result_text handles the net-cash/N-M special cases too --
        # this is the SAME function the calculation bubble's result line
        # calls, so the tile and bubble can never show different text for
        # the same DerivedMetric.
        value_text = _metric_result_text(metric)

        # Calculated values render in the default high-contrast text color
        # (no blue -- it hurt readability); the bold weight + full size they
        # already inherit from .metric-value is what distinguishes them from
        # the muted .metric-na state above. An estimated value
        # (DerivedMetric.estimated, e.g. the netted-interest-expense
        # coverage fallback) still gets a distinct goldenrod shade
        # (.metric-value-estimated, matching the "Estimated" badge below) so
        # a reader can't mistake an approximation for a clean GAAP figure at
        # a glance, before ever reaching the badge text.
        value_class = "metric-value-estimated" if metric.estimated else "metric-value-calc"

        citation_html = _render_citations_html(metric.citations, doc_id)
        body = f'<div class="metric-value"><span class="{value_class}">{value_text}</span></div>'
        detail = f'<div class="metric-citation">{citation_html}</div>' if citation_html else ""
        if metric.estimated:
            detail += '<div class="metric-estimate">Estimated -- see note</div>'
        if metric.reason:
            detail += f'<div class="metric-reason">{metric.reason}</div>'

    title_div = (
        f'<div class="metric-title">{title}'
        f'<span class="metric-info" aria-hidden="true">ⓘ</span></div>'
    )

    # The header (title + value + citation/reason) sits inside <summary> so
    # it stays visible whether the tile is expanded or not -- only the
    # explanation below is disclosure content, hidden until clicked. Native
    # <details>/<summary>: no JS, click-to-expand, accessible by default,
    # survives the Gradio HTML component's sanitizer. Citation links live
    # inside <summary> too (in `detail`), so app.py's delegated click
    # listener calls preventDefault() on a citation-link click -- otherwise
    # a citation click would also toggle the tile open/closed, since any
    # click inside a <summary> triggers the native details-toggle by default.
    explanation_text = METRIC_EXPLANATIONS.get(metric.name, "")
    explanation_body = _render_calculation_html(metric, doc_id)
    explanation = (
        f'<div class="metric-explanation">'
        f'<p>{explanation_text}</p>{explanation_body}</div>'
        if explanation_text or explanation_body
        else ""
    )
    return (
        f'<details class="metric-tile">'
        f"<summary>{title_div}{body}{detail}</summary>"
        f"{explanation}</details>"
    )


def render_group_html(title: str, metrics: dict[str, DerivedMetric], doc_id: str) -> str:
    tiles = "".join(render_metric_tile_html(m, doc_id) for m in metrics.values())
    return f'<div class="metric-group"><h3>{title}</h3><div class="metric-grid">{tiles}</div></div>'


def render_dashboard_html(
    derived: dict[str, dict[str, DerivedMetric]], fiscal_year: int, doc_id: str
) -> str:
    """doc_id is baked directly into each citation link's data-doc attribute
    at render time -- once this HTML string is built (and cached, see
    app.process_metrics/metrics_cache), it never gets mutated in place, so a
    cached dashboard's citation links keep resolving to the document they
    were extracted from even after a different document becomes the active
    session (unlike the Q&A companion controls, which read doc_id live from
    session state and so needed an explicit reset on document replace).
    """
    sections = "".join(
        render_group_html(GROUP_TITLES[key], derived[key], doc_id)
        for key in GROUP_ORDER
        if key in derived
    )
    return (
        f'{_DASHBOARD_STYLE}<div class="metrics-dashboard">'
        f"<h2>Credit Metrics — FY{fiscal_year}</h2>{sections}</div>"
    )
