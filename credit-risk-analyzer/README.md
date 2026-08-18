# Credit Risk Analyzer (RAG)

[![CI](https://github.com/srikala-g/credit-risk-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/srikala-g/credit-risk-analyzer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Upload a financial document — a 10-K, a bond prospectus, an earnings transcript
— and either **ask it natural-language questions** with page-cited, grounded
answers, or **extract a full credit-metrics dashboard** (leverage, coverage,
liquidity, cash flow) computed from the filing's own reported numbers.

Built with a Retrieval-Augmented Generation (RAG) pipeline for Q&A: the app
retrieves the most relevant passages from the uploaded document and asks a
language model to answer using only those passages, citing the pages it used.
Credit metrics use a separate, deliberately non-RAG path (see
[Design](#design)) — the model only reads off raw reported figures; every
ratio is computed in Python, never by the LLM.

> **Live demo:** _add your Hugging Face Space URL here_

---

## Features

- Upload a text-based financial PDF and ask questions about it.
- Answers are grounded in the document and cite the source page(s).
- Says when an answer is not in the document, instead of fabricating one.
- Extracts a full credit-metrics dashboard (profitability, leverage, coverage,
  liquidity, cash flow, capital return) plus a debt-maturity-wall chart — see
  [Credit Metrics](#credit-metrics) below.
- Every derived metric carries a citation back to its source line item(s); a
  metric that can't be computed is typed as "not available" (data wasn't
  disclosed) or "not applicable" (doesn't structurally apply to this filing),
  each with a reason, never silently computed with a substituted zero.
- Simple web UI with conversation history.
- Runs locally and deploys to Hugging Face Spaces.

## Product requirements

Full spec in [`docs/REQUIREMENTS.md`](docs/REQUIREMENTS.md). Summary:

- **Purpose** — let a research/credit analyst upload a filing and get answers
  grounded in that document, plus a computed credit-metrics dashboard, instead
  of relying on a model's memory or doing arithmetic by hand.
- **Primary users** — a credit/research analyst who needs to locate and cite
  facts in long filings quickly; a secondary reader is anyone evaluating this
  as a portfolio piece.
- **MVP scope** — single-document upload and Q&A per session, text-based PDFs
  only, grounded/cited answers, credit-metrics extraction with Python-computed
  ratios. Multi-document comparison, OCR for scanned PDFs, and non-English
  documents are explicitly out of scope for v1 (see the roadmap below).
- **Key non-functional requirements** — grounding/accuracy is the top priority
  (every answer and metric must trace to a source page); errors are always
  shown as friendly messages, never a stack trace; models, chunking, and top-k
  are configurable in one place (`config.py`).

## Credit metrics

Triggered by the "Extract Credit Metrics" tab, independently of Q&A. The model
extracts only **raw, as-reported figures** (with fiscal year, source statement,
and page) — every ratio below is then computed in plain Python from those raw
figures, never by the LLM. The catalog is generic across filings that report
differently, and every gap is typed rather than a single flat "missing" flag:

- **Not available** — a required line item isn't disclosed in this particular
  filing (e.g. interest expense not separately reported).
- **Not applicable** — the metric doesn't structurally apply to this filing
  (e.g. services mix % when the filing reports no product/services or segment
  revenue split), or the computation is undefined for the reported values
  (e.g. a zero denominator).

Either way, the metric is never computed with a substituted zero or a guessed
value — a wrong number is worse than a missing one in a credit tool.

**Profitability**
| Metric | What it means |
|---|---|
| Gross margin % | Revenue left after cost of goods sold, as a % of revenue. |
| Operating margin % | Profit after operating expenses, before interest and tax, as a % of revenue. |
| Net margin % | Bottom-line profit as a % of revenue. |
| Services mix % | Share of revenue coming from services vs. products, where the filing reports that split. Not applicable for filings without a product/services (or similar) revenue breakdown. |
| Revenue YoY growth % | Revenue growth vs. the prior fiscal year. |
| Services revenue growth % | Services-revenue growth vs. the prior fiscal year, where that split is reported. Not applicable otherwise. |

**Leverage & net debt**
| Metric | What it means |
|---|---|
| Total debt | All interest-bearing borrowings: current + noncurrent term debt + commercial paper. |
| Cash & securities | Liquid assets available to offset debt: cash + current + noncurrent marketable securities. |
| Net debt | Total debt minus cash & securities; negative means a **net cash** position. |
| EBITDA | Operating income + depreciation & amortization — a proxy for operating cash-generating capacity. |
| Gross debt / EBITDA | Years of EBITDA needed to repay all debt — a core leverage multiple lenders watch. |
| Net debt / EBITDA | Same, netting out cash; a net-cash position is labeled "N/M (net cash)," not shown as a negative multiple. |

**Interest coverage**
| Metric | What it means |
|---|---|
| EBITDA / interest expense | How many times over operating earnings cover interest obligations; low values signal debt-service stress. Falls back to a labeled estimate (interest payable within 12 months, from the debt note) when interest expense is netted elsewhere in the filing rather than disclosed separately. |

**Liquidity**
| Metric | What it means |
|---|---|
| Current ratio | Current assets / current liabilities — ability to cover near-term obligations. |
| Quick ratio | Same, excluding inventory (the least liquid current asset). |
| Days sales outstanding | Average days to collect receivables. |
| Days inventory outstanding | Average days inventory sits before being sold. |
| Days payable outstanding | Average days taken to pay suppliers. |
| Cash conversion cycle | DSO + DIO − DPO: net days cash is tied up in operations. Can be negative — a sign of favorable supplier/customer terms, not distress. |

**Cash flow**
| Metric | What it means |
|---|---|
| Free cash flow | Operating cash flow minus capex — cash available for debt service and shareholder returns. |
| FFO (funds from operations) | Net income + D&A + share-based comp — a cash-oriented earnings measure common in credit analysis. |
| FFO / debt % | A standard rating-agency cash-flow-coverage-of-leverage metric. |
| FCF / debt % | Free cash flow relative to total debt — roughly how fast debt could be repaid from discretionary cash flow. |

**Capital return**
| Metric | What it means |
|---|---|
| Capital return total | Dividends paid + share repurchases — total cash returned to shareholders. |
| Capital return vs. FCF % | Share of free cash flow being returned to shareholders vs. retained or used for debt paydown. |

**Debt maturity wall** — a bar chart of principal due by future calendar year,
read directly from the term-debt note, so a viewer can see when refinancing
risk concentrates.

## Design

Full diagrams and design rationale in
[`docs/architecture.md`](docs/architecture.md). Summary:

- **Two independent paths, one app.** Q&A is retrieval-augmented generation:
  parse → chunk → embed → store once at upload time, then embed the question →
  retrieve top-k → grounded, cited answer at query time. Credit metrics is a
  deliberate **second path**, not a reuse of RAG: the whole filing (page-tagged)
  goes to the model in one long-context call, because ratios need every
  relevant figure visible at once, not top-k similarity-retrieved snippets.
- **Grounding is enforced in the prompt**, not just hoped for — the system
  prompt constrains the Q&A model to answer only from retrieved context and to
  say when it can't find the answer.
- **The LLM extracts, Python computes.** The metrics-extraction prompt forbids
  the model from computing any ratio or sum; `metrics.py` computes every
  derived figure, with strict null-propagation (see above) so a missing input
  never gets silently replaced by zero.
- **A transparent in-memory vector store** (`vectorstore.py`, brute-force
  cosine similarity) — fast and dependency-free for a single document, and a
  deliberate seam to swap in FAISS/Chroma/pgvector when scaling to many
  documents.
- **Page provenance carried end-to-end** — every chunk is stamped with its
  source page at parse time, and that tag survives through embedding,
  retrieval, and into the final answer's citation.
- **Retry-aware Gemini calls.** `retry.py` distinguishes a transient per-minute
  rate limit (retry with the server-suggested delay) from a hard per-day quota
  exhaustion (fail fast with a clear message instead of stalling on retries
  that can't succeed).

## Quickstart

```bash
# clone and enter
git clone https://github.com/srikala-g/credit-risk-analyzer.git
cd credit-risk-analyzer

# set up an environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# add your API key
cp .env.example .env         # then edit .env and paste your Gemini key

# run
python app.py                # opens http://127.0.0.1:7860
```

Run the tests (no API key needed):

```bash
pip install -e ".[dev]"
pytest -q
```

## Project structure

```
credit-risk-analyzer/
├── app.py                    # Gradio web interface (entrypoint)
├── src/credit_risk_analyzer/          # application package
│   ├── config.py             # models, chunking, top-k, grounding prompt
│   ├── ingest.py             # PDF -> page text -> page-tagged chunks
│   ├── embeddings.py         # text -> vectors (swappable provider)
│   ├── vectorstore.py        # in-memory cosine-similarity search
│   ├── rag.py                # retrieve -> prompt -> cited answer
│   ├── metrics.py            # long-context line-item extraction + Python ratio engine
│   └── retry.py              # rate-limit/server-error retry with fail-fast on hard quota
├── tests/                    # pure-Python tests (no API key required)
├── docs/
│   ├── REQUIREMENTS.md       # functional & non-functional requirements
│   └── architecture.md       # design + diagrams (renders on GitHub)
├── data/sample/              # place sample PDFs here (git-ignored)
├── pyproject.toml            # packaging, dependencies, tooling config
├── requirements.txt          # runtime deps (used by Hugging Face Spaces)
├── Makefile                  # make install | dev | lint | format | test | run
└── .github/workflows/ci.yml  # lint, format check, test matrix
```

## Deploy to Hugging Face Spaces

1. Create a new **Gradio** Space.
2. Push this repository to it.
3. In **Settings → Secrets**, add `GEMINI_API_KEY`.

The Space installs `requirements.txt` and runs `app.py`.

## Development

```bash
make dev       # install package + dev tools + pre-commit hooks
make format    # auto-format (ruff)
make lint      # lint (ruff)
make test      # run tests
```

## Roadmap

- Multi-document support with per-source attribution
- OCR fallback for scanned PDFs
- Highlight the exact source sentence, not just the page
- Optional local embedding model to run fully offline
- Retrieval/answer evaluation set

## License

Released under the [MIT License](LICENSE).

---

_This project is for research and educational use. It is not financial advice._
