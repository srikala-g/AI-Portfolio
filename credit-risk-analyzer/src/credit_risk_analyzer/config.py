"""Central configuration for the Credit Risk Analyzer app.

All tunable knobs live here so the rest of the code stays clean and so you can
talk through your design choices in an interview ("I made chunk size and top-k
configurable because...").
"""

import os

# Load variables from a local .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; env vars can be set directly
    pass

# ---- Models ---------------------------------------------------------------
# Embedding model turns text into vectors we can compare by meaning. By default
# we use a local sentence-transformers model so indexing does not require any
# embedding API calls.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Chat model writes the final grounded answer from the retrieved context.
# gemini-2.5-flash is fast, cheap, and available on the free tier.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# Local embeddings are encoded in batches for efficiency.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# ---- Chunking -------------------------------------------------------------
# Documents are split into overlapping chunks. Overlap keeps sentences that
# straddle a boundary from being lost. Larger chunks mean fewer chunks overall,
# which means fewer embedding calls — helpful under the free-tier rate limit.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))  # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # characters shared

# ---- Embedding cache ------------------------------------------------------
# Cache embeddings to disk so large PDFs are embedded once per compatible
# configuration.
CACHE_DIR = os.getenv("CACHE_DIR", ".cache/embeddings")

# ---- Metrics extraction cache ----------------------------------------------
# Cache the raw Gemini extraction response to disk, keyed by the exact prompt
# sent. Re-analyzing the same filing (e.g. rehearsing a demo) then costs zero
# API calls and returns instantly instead of re-running the full-filing call
# and risking the free-tier rate limit again.
METRICS_CACHE_DIR = os.getenv("METRICS_CACHE_DIR", ".cache/metrics")

# ---- Page image cache (citation viewer) -------------------------------------
# Clicking a citation rasterizes the source PDF page it points to. Rendered
# pages are cached to disk keyed by (doc_id, page) -- same spirit as the
# embedding and metrics-extraction caches above -- so clicking the same
# citation twice is instant instead of re-rasterizing every time.
PAGE_IMAGE_CACHE_DIR = os.getenv("PAGE_IMAGE_CACHE_DIR", ".cache/page_images")
PAGE_IMAGE_DPI = int(os.getenv("PAGE_IMAGE_DPI", "200"))

# ---- Retrieval ------------------------------------------------------------
# How many chunks to retrieve and feed to the model as context.
TOP_K = int(os.getenv("TOP_K", "4"))

# ---- Generation -----------------------------------------------------------
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))  # 0 = factual, deterministic

# The system prompt is where grounding + citation behaviour is enforced.
SYSTEM_PROMPT = (
    "You are a careful financial research assistant. Answer the user's question "
    "using ONLY the provided context excerpts. If the answer is not in the "
    "context, say you cannot find it in the document rather than guessing. "
    "Cite the page number(s) you used in square brackets, e.g. [p. 4]. "
    "Be concise and precise with figures."
)

# ---- Credit metrics extraction ---------------------------------------------
# Raw line-item extraction uses a long-context call over the parsed filing text
# instead of the embedding/RAG path, so it isn't subject to the embedding
# rate limit on large filings. Kept as its own model/temperature knob so it can
# be tuned independently of the chat model above.
METRICS_MODEL = os.getenv("METRICS_MODEL", "gemini-2.5-flash")
METRICS_TEMPERATURE = float(os.getenv("METRICS_TEMPERATURE", "0.0"))

# The extraction prompt is deliberately restricted to raw reported figures: the
# model must never compute a ratio itself. All derived metrics are computed in
# Python (see metrics.py) so a wrong number is a bug we can find, not a
# hallucinated computation.
METRICS_SYSTEM_PROMPT = (
    "You are a meticulous financial-statement data-entry assistant. Extract ONLY "
    "raw, as-reported line items from the filing text below — never compute a "
    "ratio, percentage, sum, or any derived figure yourself. For every line item "
    "you find, report exactly the number as printed, the fiscal year it belongs "
    "to, the statement or note it came from, and the page number tag (e.g. "
    "'[p. 12]') it appeared under. If a line item is not present in the text, "
    "set its value and page to null rather than guessing or estimating. "
    "Respond with STRICT JSON only: no prose, no explanation, no markdown code "
    "fences."
)
