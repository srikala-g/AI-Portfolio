"""Credit Risk Analyzer — a retrieval-augmented Q&A app over financial PDFs.

Uses lazy attribute loading (PEP 562) so that importing the package — or just
the ingestion/vectorstore pieces — does not pull in the LLM client or require an
API key. The key is only needed when you actually embed or generate.
"""

__all__ = [
    "build_index",
    "answer_question",
    "Answer",
    "VectorStore",
    "extract_line_items",
    "compute_all_derived_metrics",
    "ExtractedFinancials",
]


def __getattr__(name):
    if name in {"build_index", "answer_question", "Answer"}:
        from . import rag

        return getattr(rag, name)
    if name == "VectorStore":
        from .vectorstore import VectorStore

        return VectorStore
    if name in {"extract_line_items", "compute_all_derived_metrics", "ExtractedFinancials"}:
        from . import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
