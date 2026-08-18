"""Tests for credit metrics: derived-ratio math + defensive JSON parsing.

Run without any API key or network access:
- The derived-metrics tests feed a fixed dict of line items (real figures read
  from tests/fixtures/apple-10k-2025.pdf, FY2025 + FY2024) straight into the
  ratio functions -- no Gemini call involved.
- The extraction test mocks the Gemini client so `extract_line_items` never
  hits the network.
"""

import json

import pytest

import credit_risk_analyzer.metrics as metrics
from credit_risk_analyzer.metrics import (
    Citation,
    DerivedMetric,
    ExtractedFinancials,
    InputTrace,
    LineItem,
    MaturityYear,
    UnavailableReason,
    compute_all_derived_metrics,
    compute_coverage,
    compute_leverage,
    compute_liquidity,
    debt_maturity_chart_data,
    format_input_value,
    format_value,
    parse_llm_response,
    render_dashboard_html,
    render_metric_tile_html,
)

FY = 2025
PRIOR_FY = 2024

# Real figures read from tests/fixtures/apple-10k-2025.pdf (page numbers are the
# PDF's own reading-order page index, matching how ingest.load_pdf_pages numbers
# pages elsewhere in this codebase).
INCOME_STATEMENT = "Consolidated Statements of Operations"
BALANCE_SHEET = "Consolidated Balance Sheets"
CASH_FLOW_STATEMENT = "Consolidated Statements of Cash Flows"
DEBT_NOTE = "Notes to Financial Statements - Term Debt"
MDA_DEBT = "MD&A - Contractual Obligations (Debt)"

# name -> fiscal_year -> (value, unit, source_statement, page)
RAW_LINE_ITEMS: dict[str, dict[int, tuple]] = {
    "total_revenue": {
        FY: (416161, "USD_MILLIONS", INCOME_STATEMENT, 33),
        PRIOR_FY: (391035, "USD_MILLIONS", INCOME_STATEMENT, 33),
    },
    "products_revenue": {FY: (307003, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "services_revenue": {
        FY: (109158, "USD_MILLIONS", INCOME_STATEMENT, 33),
        PRIOR_FY: (96169, "USD_MILLIONS", INCOME_STATEMENT, 33),
    },
    "cost_of_sales": {FY: (220960, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "gross_margin": {FY: (195201, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "operating_income": {FY: (133050, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "net_income": {FY: (112010, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "research_and_development": {FY: (34550, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "selling_general_administrative": {FY: (27601, "USD_MILLIONS", INCOME_STATEMENT, 33)},
    "depreciation_amortization": {FY: (11698, "USD_MILLIONS", CASH_FLOW_STATEMENT, 37)},
    "share_based_comp": {FY: (12863, "USD_MILLIONS", CASH_FLOW_STATEMENT, 37)},
    # Apple nets interest expense into "Other income/(expense), net" -- no clean
    # interest_expense line, so it's null and we fall back to the debt-note estimate.
    "interest_expense": {FY: (None, None, None, None)},
    "interest_expense_estimate": {FY: (2600, "USD_MILLIONS", MDA_DEBT, 29)},
    "cash_and_equivalents": {FY: (35934, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "current_marketable_securities": {FY: (18763, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "noncurrent_marketable_securities": {FY: (77723, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "total_current_assets": {FY: (147957, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "total_current_liabilities": {FY: (165631, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "inventories": {FY: (5718, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "accounts_receivable": {FY: (39777, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "accounts_payable": {FY: (69860, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "current_term_debt": {FY: (12350, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "noncurrent_term_debt": {FY: (78328, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "commercial_paper": {FY: (7979, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "total_assets": {FY: (359241, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "total_liabilities": {FY: (285508, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "total_shareholders_equity": {FY: (73733, "USD_MILLIONS", BALANCE_SHEET, 35)},
    "operating_cash_flow": {FY: (111482, "USD_MILLIONS", CASH_FLOW_STATEMENT, 37)},
    "capex": {FY: (12715, "USD_MILLIONS", CASH_FLOW_STATEMENT, 37)},
    "dividends_paid": {FY: (15421, "USD_MILLIONS", CASH_FLOW_STATEMENT, 37)},
    "share_repurchases": {FY: (90711, "USD_MILLIONS", CASH_FLOW_STATEMENT, 37)},
}

MATURITY_SCHEDULE = [
    (2026, 12393),
    (2027, 10078),
    (2028, 9300),
    (2029, 5235),
    (2030, 4972),
]


def _build_financials(
    raw: dict[str, dict[int, tuple]] | None = None,
    schedule: list[tuple[int, int]] | None = None,
    fiscal_years: list[int] | None = None,
) -> ExtractedFinancials:
    raw = RAW_LINE_ITEMS if raw is None else raw
    schedule = MATURITY_SCHEDULE if schedule is None else schedule
    line_items: dict[str, dict[int, LineItem]] = {}
    for name, by_year in raw.items():
        line_items[name] = {
            fy: LineItem(value=value, unit=unit, fiscal_year=fy, source_statement=stmt, page=page)
            for fy, (value, unit, stmt, page) in by_year.items()
        }
    maturities = [
        MaturityYear(
            year=year, value=value, unit="USD_MILLIONS", source_statement=DEBT_NOTE, page=50
        )
        for year, value in schedule
    ]
    return ExtractedFinancials(
        fiscal_years=fiscal_years or [FY, PRIOR_FY],
        line_items=line_items,
        debt_maturity_schedule=maturities,
    )


@pytest.fixture
def financials() -> ExtractedFinancials:
    return _build_financials()


# ---------------------------------------------------------------------------
# Derived metrics against known Apple FY2025 targets
# ---------------------------------------------------------------------------


def test_profitability_targets(financials):
    derived = compute_all_derived_metrics(financials, FY, PRIOR_FY)
    profitability = derived["profitability"]

    assert profitability["gross_margin_pct"].value == pytest.approx(46.9, abs=0.1)
    assert profitability["operating_margin_pct"].value == pytest.approx(32.0, abs=0.1)
    assert profitability["net_margin_pct"].value == pytest.approx(26.9, abs=0.1)
    assert profitability["revenue_yoy_growth_pct"].value == pytest.approx(6.4, abs=0.1)

    # All inputs present (including the revenue breakdown) -> computes normally,
    # not flagged unavailable under either typed reason.
    assert profitability["services_mix_pct"].value == pytest.approx(26.2, abs=0.1)
    assert profitability["services_mix_pct"].unavailable_reason is None
    assert profitability["services_revenue_growth_pct"].value == pytest.approx(13.5, abs=0.1)
    assert profitability["services_revenue_growth_pct"].unavailable_reason is None


def test_leverage_targets(financials):
    leverage = compute_all_derived_metrics(financials, FY)["leverage"]

    assert leverage["total_debt"].value == pytest.approx(98_657, abs=1)
    assert leverage["cash_and_securities"].value == pytest.approx(132_420, abs=1)
    assert leverage["net_debt"].value == pytest.approx(-33_763, abs=1)
    assert leverage["ebitda"].value == pytest.approx(144_748, abs=1)
    assert leverage["gross_debt_to_ebitda"].value == pytest.approx(0.68, abs=0.01)


def test_net_debt_is_negative_meaning_net_cash(financials):
    """Apple has more cash+securities than debt -- net_debt must be negative,
    and the leverage multiple built from it should not be rendered as if it
    were meaningful positive leverage."""
    leverage = compute_all_derived_metrics(financials, FY)["leverage"]
    assert leverage["net_debt"].value < 0
    assert leverage["net_debt_to_ebitda"].value < 0

    tile_html = render_metric_tile_html(leverage["net_debt"], "doc-a")
    assert "Net cash" in tile_html
    assert "-33" not in tile_html  # never shown as a bare negative number

    multiple_html = render_metric_tile_html(leverage["net_debt_to_ebitda"], "doc-a")
    assert "N/M" in multiple_html


def test_cash_flow_targets(financials):
    cash_flow = compute_all_derived_metrics(financials, FY)["cash_flow"]

    assert cash_flow["free_cash_flow"].value == pytest.approx(98_767, abs=1)
    assert cash_flow["ffo"].value == pytest.approx(136_571, abs=1)


def test_liquidity_targets(financials):
    liquidity = compute_liquidity(financials.line_items, FY)

    assert liquidity["current_ratio"].value == pytest.approx(0.89, abs=0.01)
    # Negative cash conversion cycle -- confirms the sub-1.0x current ratio
    # isn't automatically distress; there's no hardcoded threshold to misfire.
    assert liquidity["cash_conversion_cycle"].value < 0


# ---------------------------------------------------------------------------
# Edge cases: netted interest expense, missing inputs, division by zero
# ---------------------------------------------------------------------------


def test_coverage_falls_back_to_estimate_when_interest_expense_is_netted(financials):
    leverage = compute_leverage(financials.line_items, FY)
    coverage = compute_coverage(financials.line_items, FY, leverage["ebitda"])
    metric = coverage["ebitda_to_interest_expense"]

    assert metric.value == pytest.approx(144_748 / 2600, rel=1e-6)
    assert metric.estimated is True
    assert metric.reason is not None and "estimate" in metric.reason.lower()


def test_coverage_is_not_available_when_no_interest_data_at_all():
    excluded = ("interest_expense", "interest_expense_estimate")
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k not in excluded}
    fin = _build_financials(raw=raw, fiscal_years=[FY])
    leverage = compute_leverage(fin.line_items, FY)
    coverage = compute_coverage(fin.line_items, FY, leverage["ebitda"])
    metric = coverage["ebitda_to_interest_expense"]

    assert metric.value is None
    assert metric.reason is not None and "not available" in metric.reason.lower()
    assert metric.unavailable_reason == UnavailableReason.INPUT_MISSING


def test_missing_debt_component_nulls_total_debt_and_cascades():
    """Per project policy: a missing raw input must never be treated as 0 --
    it must flag the derived metric (and everything downstream) as not
    available (INPUT_MISSING), not silently compute a partial sum."""
    raw = dict(RAW_LINE_ITEMS)
    raw = {k: v for k, v in raw.items() if k != "commercial_paper"}
    fin = _build_financials(raw=raw, fiscal_years=[FY])

    leverage = compute_leverage(fin.line_items, FY)
    assert leverage["total_debt"].value is None
    assert "commercial_paper" in leverage["total_debt"].reason
    assert leverage["total_debt"].unavailable_reason == UnavailableReason.INPUT_MISSING

    # Cascades: net_debt and the leverage multiples depend on total_debt.
    assert leverage["net_debt"].value is None
    assert leverage["gross_debt_to_ebitda"].value is None
    assert "total_debt" in leverage["gross_debt_to_ebitda"].reason
    assert leverage["gross_debt_to_ebitda"].unavailable_reason == UnavailableReason.INPUT_MISSING

    # And cash-flow metrics that depend on total_debt cascade too. No metric
    # anywhere in the cascade computes a value -- a missing component is never
    # silently treated as 0 in the sum.
    cash_flow = compute_all_derived_metrics(fin, FY)["cash_flow"]
    assert cash_flow["ffo_to_debt_pct"].value is None
    assert cash_flow["fcf_to_debt_pct"].value is None


def test_division_by_zero_returns_none_not_crash():
    raw = dict(RAW_LINE_ITEMS)
    raw["total_current_liabilities"] = {FY: (0, "USD_MILLIONS", BALANCE_SHEET, 35)}
    fin = _build_financials(raw=raw, fiscal_years=[FY])

    liquidity = compute_liquidity(fin.line_items, FY)
    assert liquidity["current_ratio"].value is None
    assert "division by zero" in liquidity["current_ratio"].reason
    # Inputs were present (liabilities reported as exactly 0) -- this is a
    # structurally undefined computation, not missing data.
    assert liquidity["current_ratio"].unavailable_reason == UnavailableReason.NOT_APPLICABLE


def test_missing_line_item_produces_reason_not_crash():
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k != "total_revenue"}
    fin = _build_financials(raw=raw, fiscal_years=[FY])

    profitability = compute_all_derived_metrics(fin, FY)["profitability"]
    assert profitability["gross_margin_pct"].value is None
    assert profitability["gross_margin_pct"].reason is not None
    assert profitability["gross_margin_pct"].unavailable_reason == UnavailableReason.INPUT_MISSING


def test_services_metrics_not_applicable_without_revenue_breakdown():
    """A filing with no product/services (or similar) revenue split -- the
    common case, not just Apple's -- must not crash or silently compute a
    services metric. It resolves to NOT_APPLICABLE, distinct from a metric
    whose data just wasn't found, while the rest of the profitability group
    (which doesn't depend on a revenue breakdown) still computes normally."""
    excluded = ("products_revenue", "services_revenue")
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k not in excluded}
    fin = _build_financials(raw=raw, fiscal_years=[FY, PRIOR_FY])

    profitability = compute_all_derived_metrics(fin, FY, PRIOR_FY)["profitability"]

    services_mix = profitability["services_mix_pct"]
    assert services_mix.value is None
    assert services_mix.unavailable_reason == UnavailableReason.NOT_APPLICABLE
    assert "not applicable" in services_mix.reason.lower()

    services_growth = profitability["services_revenue_growth_pct"]
    assert services_growth.value is None
    assert services_growth.unavailable_reason == UnavailableReason.NOT_APPLICABLE
    assert "not applicable" in services_growth.reason.lower()

    # Genericity: metrics that don't depend on a revenue breakdown are
    # unaffected by its absence.
    assert profitability["gross_margin_pct"].value == pytest.approx(46.9, abs=0.1)
    assert profitability["net_margin_pct"].value == pytest.approx(26.9, abs=0.1)


def test_services_metric_missing_only_for_one_year_is_input_missing_not_not_applicable():
    """The filing DOES report the revenue breakdown (services_revenue exists
    for other years) -- a gap in just one year is a real data gap
    (INPUT_MISSING), not a structural absence (NOT_APPLICABLE)."""
    raw = dict(RAW_LINE_ITEMS)
    raw["services_revenue"] = {PRIOR_FY: raw["services_revenue"][PRIOR_FY]}  # drop FY value
    fin = _build_financials(raw=raw, fiscal_years=[FY, PRIOR_FY])

    profitability = compute_all_derived_metrics(fin, FY, PRIOR_FY)["profitability"]
    services_mix = profitability["services_mix_pct"]

    assert services_mix.value is None
    assert services_mix.unavailable_reason == UnavailableReason.INPUT_MISSING


def test_render_metric_tile_html_distinguishes_not_applicable_from_not_available():
    input_missing = DerivedMetric(
        name="gross_margin_pct",
        value=None,
        unit="%",
        formula="gross_margin / total_revenue * 100",
        fiscal_year=FY,
        reason="not available: total_revenue (not extracted from filing)",
        unavailable_reason=UnavailableReason.INPUT_MISSING,
    )
    not_applicable = DerivedMetric(
        name="services_mix_pct",
        value=None,
        unit="%",
        formula="services_revenue / total_revenue * 100",
        fiscal_year=FY,
        reason="not applicable: this filing does not report a product/services breakdown",
        unavailable_reason=UnavailableReason.NOT_APPLICABLE,
    )

    assert "Not available" in render_metric_tile_html(input_missing, "doc-a")
    assert "Not applicable" in render_metric_tile_html(not_applicable, "doc-a")
    assert "Not applicable" not in render_metric_tile_html(input_missing, "doc-a")
    assert "Not available" not in render_metric_tile_html(not_applicable, "doc-a")


def test_debt_maturity_chart_data_shape(financials):
    df = debt_maturity_chart_data(financials)
    assert list(df.columns) == ["year", "principal"]
    assert len(df) == len(MATURITY_SCHEDULE)
    assert df.iloc[0]["year"] == "2026"
    assert df.iloc[0]["principal"] == 12393


# ---------------------------------------------------------------------------
# Defensive JSON parsing (no network)
# ---------------------------------------------------------------------------


def test_parse_llm_response_strips_markdown_fences():
    payload = {
        "fiscal_years": [2025],
        "line_items": {
            "total_revenue": [
                {
                    "fiscal_year": 2025,
                    "value": 416161,
                    "unit": "USD_MILLIONS",
                    "source_statement": INCOME_STATEMENT,
                    "page": 33,
                }
            ]
        },
        "debt_maturity_schedule": [],
    }
    fenced = f"```json\n{json.dumps(payload)}\n```"

    result = parse_llm_response(fenced)

    assert result.fiscal_years == [2025]
    assert result.line_items["total_revenue"][2025].value == 416161
    assert result.line_items["total_revenue"][2025].page == 33
    # Every other declared key is filled in as an empty (not missing) mapping.
    assert result.line_items["interest_expense"] == {}


def test_parse_llm_response_rejects_invalid_json():
    with pytest.raises(ValueError):
        parse_llm_response("not json at all")


# ---------------------------------------------------------------------------
# Extraction with a mocked Gemini client (no network)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text


class _FakeModels:
    def __init__(self, text: str):
        self._text = text

    def generate_content(self, model, contents, config):
        return _FakeResponse(self._text)


class _FakeClient:
    def __init__(self, text: str):
        self.models = _FakeModels(text)


_EXTRACT_PAYLOAD = {
    "fiscal_years": [2025],
    "line_items": {
        "total_revenue": [
            {
                "fiscal_year": 2025,
                "value": 416161,
                "unit": "USD_MILLIONS",
                "source_statement": INCOME_STATEMENT,
                "page": 33,
            }
        ]
    },
    "debt_maturity_schedule": [
        {
            "year": 2026,
            "value": 12393,
            "unit": "USD_MILLIONS",
            "source_statement": DEBT_NOTE,
            "page": 50,
        }
    ],
}


def test_extract_line_items_uses_mocked_client_not_network(tmp_path, monkeypatch):
    monkeypatch.setattr(metrics, "METRICS_CACHE_DIR", str(tmp_path / "metrics-cache"))
    fenced_response = f"```json\n{json.dumps(_EXTRACT_PAYLOAD)}\n```"
    monkeypatch.setattr(metrics, "_get_client", lambda: _FakeClient(fenced_response))

    result = metrics.extract_line_items("tests/fixtures/apple-10k-2025.pdf")

    assert result.fiscal_years == [2025]
    assert result.line_items["total_revenue"][2025].value == 416161
    assert result.debt_maturity_schedule[0].year == 2026


def test_extract_line_items_cache_hit_skips_client(tmp_path, monkeypatch):
    """Second call for the same filing must not touch the client at all."""
    monkeypatch.setattr(metrics, "METRICS_CACHE_DIR", str(tmp_path / "metrics-cache"))
    fenced_response = f"```json\n{json.dumps(_EXTRACT_PAYLOAD)}\n```"
    monkeypatch.setattr(metrics, "_get_client", lambda: _FakeClient(fenced_response))

    first = metrics.extract_line_items("tests/fixtures/apple-10k-2025.pdf")

    def _boom():
        raise AssertionError("cache hit should not call _get_client")

    monkeypatch.setattr(metrics, "_get_client", _boom)
    second = metrics.extract_line_items("tests/fixtures/apple-10k-2025.pdf")

    assert second.fiscal_years == first.fiscal_years
    assert second.line_items["total_revenue"][2025].value == 416161


def test_render_metric_tile_html_shows_estimate_caveat():
    metric = DerivedMetric(
        name="ebitda_to_interest_expense",
        value=55.7,
        unit="x",
        formula="ebitda / interest_expense_estimate",
        fiscal_year=FY,
        citations=[Citation(MDA_DEBT, 29)],
        reason="estimate: interest expense is netted elsewhere...",
        estimated=True,
    )
    html = render_metric_tile_html(metric, "doc-a")
    assert "Estimated" in html
    assert "p.29" in html


# ---------------------------------------------------------------------------
# In-text citation links (each Citation -> its own <a class="citation-link">)
# ---------------------------------------------------------------------------


def test_citation_with_a_page_renders_as_a_clickable_link():
    metric = DerivedMetric(
        name="gross_margin_pct",
        value=46.9,
        unit="%",
        formula="gross_margin / total_revenue * 100",
        fiscal_year=FY,
        citations=[Citation(INCOME_STATEMENT, 33)],
    )
    html = render_metric_tile_html(metric, "doc-a")

    assert '<a href="#" class="citation-link"' in html
    assert 'data-doc="doc-a"' in html
    assert 'data-page="33"' in html
    assert "Consolidated Statements of Operations, p.33" in html


def test_two_citations_on_one_metric_become_two_independent_links():
    """total_debt-style metrics can cite different pages for different line
    items -- each must be its own link so clicking one opens the page it
    actually names, not the first (or an arbitrary) citation."""
    metric = DerivedMetric(
        name="ebitda",
        value=144_748,
        unit="USD_MILLIONS",
        formula="operating_income + depreciation_amortization",
        fiscal_year=FY,
        citations=[Citation(INCOME_STATEMENT, 33), Citation(CASH_FLOW_STATEMENT, 37)],
    )
    html = render_metric_tile_html(metric, "doc-a")

    # Rendered in two places (the always-visible citation line and the
    # explanation panel's "Source:" line) -- what matters is that page 33
    # and page 37 are each independently clickable, not folded into one link.
    assert html.count('data-page="33"') >= 1
    assert html.count('data-page="37"') >= 1
    assert 'data-page="33"' in html
    assert 'data-page="37"' in html


def test_unavailable_metric_produces_no_citation_link():
    """A metric with no value has no citations (per _combine) -- confirm the
    render layer doesn't invent a link for it; the NA state stays plain
    text, exactly as before this feature."""
    metric = DerivedMetric(
        name="gross_margin_pct",
        value=None,
        unit="%",
        formula="gross_margin / total_revenue * 100",
        fiscal_year=FY,
        reason="not available: total_revenue (not extracted from filing)",
        unavailable_reason=UnavailableReason.INPUT_MISSING,
    )
    html = render_metric_tile_html(metric, "doc-a")

    assert "citation-link" not in html
    assert "data-page" not in html


def test_citation_with_no_page_renders_as_plain_text_not_a_link():
    """format_citation still returns a label (the statement name) when a
    citation has no page -- that label must not become a dead link with no
    page to open."""
    metric = DerivedMetric(
        name="total_debt",
        value=98_657,
        unit="USD_MILLIONS",
        formula="current_term_debt + noncurrent_term_debt + commercial_paper",
        fiscal_year=FY,
        citations=[Citation(BALANCE_SHEET, None)],
    )
    html = render_metric_tile_html(metric, "doc-a")

    assert "citation-link" not in html
    assert BALANCE_SHEET in html


def test_dashboard_bakes_doc_id_into_every_citation_link(financials):
    """render_dashboard_html threads doc_id through to every tile's
    citations -- this is what makes a cached dashboard's links keep
    resolving to the document they were extracted from, even after a
    different document becomes the active session (see
    render_dashboard_html's docstring)."""
    derived = compute_all_derived_metrics(financials, FY, PRIOR_FY)
    html = render_dashboard_html(derived, FY, "doc-fixed-id")

    assert html.count('data-doc="doc-fixed-id"') >= 1
    assert "data-doc=\"None\"" not in html


# ---------------------------------------------------------------------------
# Calculation view in the explanation bubble: displayed inputs/result must
# be the DerivedMetric's own captured values, never recomputed or
# independently reformatted (the key drift-guard requirement).
# ---------------------------------------------------------------------------


def test_combine_traces_every_input_even_when_they_share_a_page(financials):
    """The bug this feature exists to fix: total_debt's three inputs
    (current_term_debt, noncurrent_term_debt, commercial_paper) all cite
    the same page (Consolidated Balance Sheets, p.35), so the OLD flat
    `citations` list (deduped by statement+page) collapses them to one
    entry and silently drops two of the three values. `inputs` must not
    have that loss -- each input keeps its own distinct value and citation
    even when several inputs share a page."""
    total_debt = compute_leverage(financials.line_items, FY)["total_debt"]

    assert len(total_debt.inputs) == 3
    by_label = {t.label: t for t in total_debt.inputs}
    assert by_label["current_term_debt"].value == 12_350
    assert by_label["noncurrent_term_debt"].value == 78_328
    assert by_label["commercial_paper"].value == 7_979
    # All three legitimately share a page -- confirms this is really the
    # same-page scenario the flat `citations` list would have collapsed.
    pages = {t.citation.page for t in total_debt.inputs if t.citation}
    assert pages == {35}
    statements = {t.citation.statement for t in total_debt.inputs if t.citation}
    assert statements == {"Consolidated Balance Sheets"}


def test_displayed_calculation_matches_the_derivedmetric_captured_values(financials):
    """Regression guard against display/compute drift: render a real
    computed metric's tile and assert every number that appears in the
    calculation view -- each input's value and the result -- is exactly
    what compute_leverage's DerivedMetric.inputs/value actually holds, not
    a re-derived or independently rounded figure. Walks the HTML back
    against the source of truth rather than hard-coding expected strings,
    so it fails if rendering ever diverges from what was actually computed.
    """
    total_debt = compute_leverage(financials.line_items, FY)["total_debt"]
    html = render_metric_tile_html(total_debt, "doc-a")

    # Every input's as-printed value (format_input_value: plain millions,
    # never billion-converted) appears in the calculation view.
    for trace in total_debt.inputs:
        expected = format_input_value(trace.value, trace.unit)
        assert expected in html, f"{trace.label}'s captured value {expected!r} not found in rendered HTML"
        assert trace.citation is not None
        assert f'data-page="{trace.citation.page}"' in html
        assert f'data-value="{trace.citation.value}"' in html

    # The result line matches format_value(metric.value, metric.unit) --
    # the exact same call the tile's own headline value uses -- verbatim.
    expected_result = format_value(total_debt.value, total_debt.unit)
    assert html.count(expected_result) >= 2  # once in the tile value, once in the "Result:" line


def test_calculation_result_line_uses_the_same_text_as_the_tile_value():
    """_metric_result_text is the single function both the tile's headline
    value and the calculation bubble's result line call -- assert the
    calculation view's "Result:" line and the tile's own value render
    identical text, including the net-cash special case, which is exactly
    the kind of place independent formatting could have silently diverged."""
    net_debt = DerivedMetric(
        name="net_debt",
        value=-33_763.0,
        unit="USD_MILLIONS",
        formula="total_debt - cash_and_securities",
        fiscal_year=FY,
        inputs=[
            InputTrace(label="total_debt", value=98_657.0, unit="USD_MILLIONS"),
            InputTrace(label="cash_and_securities", value=132_420.0, unit="USD_MILLIONS"),
        ],
    )
    html = render_metric_tile_html(net_debt, "doc-a")

    assert html.count("Net cash: $33.8B") == 2  # tile value + calculation result line


def test_unavailable_metric_shows_all_inputs_found_and_missing():
    """Per policy: an unavailable metric's calculation view shows every
    input, not just the failing one(s) -- both what succeeded and what
    didn't, so the cascade is legible (e.g. "current_term_debt = $12,350M"
    alongside "commercial_paper = not extracted from filing")."""
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k != "commercial_paper"}
    fin = _build_financials(raw=raw, fiscal_years=[FY])

    total_debt = compute_leverage(fin.line_items, FY)["total_debt"]
    assert total_debt.value is None

    by_label = {t.label: t for t in total_debt.inputs}
    assert set(by_label) == {"current_term_debt", "noncurrent_term_debt", "commercial_paper"}
    assert by_label["current_term_debt"].value == 12_350
    assert by_label["current_term_debt"].citation is not None
    assert by_label["noncurrent_term_debt"].value == 78_328
    assert by_label["commercial_paper"].value is None
    assert by_label["commercial_paper"].unavailable_reason == "not extracted from filing"

    html = render_metric_tile_html(total_debt, "doc-a")
    assert "current_term_debt = " in html
    assert "$12,350M" in html
    assert "commercial_paper = " in html
    assert "not extracted from filing" in html
    assert "metric-calc-input-missing" in html


def test_compute_coverage_traces_ebitda_and_the_chosen_interest_input():
    """compute_coverage builds DerivedMetric manually (it doesn't go
    through _combine), so its input-tracing is hand-written -- verify it
    independently. Apple's real case: ebitda is found, interest_expense
    isn't (netted into "Other income/(expense), net"), no estimate either
    -- both should show, one succeeded and one didn't."""
    excluded = ("interest_expense", "interest_expense_estimate")
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k not in excluded}
    fin = _build_financials(raw=raw, fiscal_years=[FY])
    leverage = compute_leverage(fin.line_items, FY)
    coverage = compute_coverage(fin.line_items, FY, leverage["ebitda"])["ebitda_to_interest_expense"]

    by_label = {t.label: t for t in coverage.inputs}
    assert by_label["ebitda"].value == leverage["ebitda"].value
    assert by_label["ebitda"].citation is None  # nested DerivedMetric input, one-level cascade
    assert by_label["interest_expense"].value is None
    assert by_label["interest_expense"].unavailable_reason is not None


def test_compute_coverage_estimate_branch_traces_the_estimated_input():
    """The successful, estimated branch: interest_expense_estimate is used
    in place of interest_expense -- confirm its InputTrace carries the
    estimate's own citation/value, labeled distinctly from the clean case."""
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k != "interest_expense"}
    fin = _build_financials(raw=raw, fiscal_years=[FY])
    leverage = compute_leverage(fin.line_items, FY)
    coverage = compute_coverage(fin.line_items, FY, leverage["ebitda"])["ebitda_to_interest_expense"]

    by_label = {t.label: t for t in coverage.inputs}
    assert "interest_expense_estimate" in by_label
    trace = by_label["interest_expense_estimate"]
    assert trace.value == 2_600
    assert trace.citation is not None
    assert trace.citation.page == 29


def test_nested_derivedmetric_input_has_no_citation_one_level_cascade():
    """net_debt_to_ebitda's inputs are themselves DerivedMetrics (net_debt,
    ebitda) -- per the one-level cascade design, their InputTrace carries a
    value but no citation (they aren't single-page figures), and the
    calculation view shows them as plain text, not a link."""
    derived = compute_all_derived_metrics(_build_financials(), FY)
    leverage = derived["leverage"]
    net_debt_to_ebitda = leverage["net_debt_to_ebitda"]

    by_label = {t.label: t for t in net_debt_to_ebitda.inputs}
    assert by_label["net_debt"].citation is None
    assert by_label["ebitda"].citation is None
    assert by_label["net_debt"].value == leverage["net_debt"].value

    html = render_metric_tile_html(net_debt_to_ebitda, "doc-a")
    assert "net_debt = " in html
    # No citation-link wrapping the net_debt input line's value specifically
    # -- it's a nested derived value, shown plain.
    assert 'data-page' not in html.split("net_debt = ")[1].split("</div>")[0]


def test_format_input_value_never_billion_converts_unlike_format_value():
    """The reconciliation fix: a value >= 1000 (USD_MILLIONS) must render
    in plain millions for an input line, matching what's boxed on the
    source page ("95,281"), even though format_value would billion-convert
    the same number for the tile ("$95.3B")."""
    assert format_input_value(95_281.0, "USD_MILLIONS") == "$95,281M"
    assert format_value(95_281.0, "USD_MILLIONS") == "$95.3B"
    # Small values and other units are unaffected -- delegate straight
    # through to format_value, which never magnitude-scales them anyway.
    assert format_input_value(500.0, "USD_MILLIONS") == format_value(500.0, "USD_MILLIONS")
    assert format_input_value(46.9, "%") == format_value(46.9, "%")
    assert format_input_value(35.0, "days") == format_value(35.0, "days")
    assert format_input_value(None, "USD_MILLIONS") == "—"


def test_not_applicable_metric_has_no_inputs_shows_formula_only():
    """_not_applicable (the structural-skip path, e.g. services_mix_pct
    when a filing reports no product/services breakdown at all) never had
    inputs to gather -- DerivedMetric.inputs stays empty, and the
    calculation view correctly shows just the formula, no input breakdown,
    no crash."""
    excluded = ("products_revenue", "services_revenue")
    raw = {k: v for k, v in RAW_LINE_ITEMS.items() if k not in excluded}
    fin = _build_financials(raw=raw, fiscal_years=[FY, PRIOR_FY])
    profitability = compute_all_derived_metrics(fin, FY, PRIOR_FY)["profitability"]
    services_mix = profitability["services_mix_pct"]

    assert services_mix.inputs == []
    html = render_metric_tile_html(services_mix, "doc-a")
    assert "metric-calc-inputs" not in html
    assert "metric-calc-result" not in html
    assert "Formula" in html  # the formula line itself still shows
