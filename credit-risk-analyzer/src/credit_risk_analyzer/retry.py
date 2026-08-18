"""Retry helper for Gemini API calls.

The free tier enforces per-minute rate limits (e.g. embedding requests). When we
exceed them the API returns HTTP 429 with a suggested retry delay. This helper
catches transient errors (429 and 5xx), waits the delay the server asks for
(falling back to exponential backoff), and retries — so indexing a large
document completes on the free tier instead of failing partway through.

A 429 can also mean a hard per-day quota has been exhausted rather than a
transient per-minute limit. Confirmed against a live 429 payload: Gemini still
attaches a short `retryDelay` (e.g. 58s) to a per-day quota error even though
retrying within that window cannot possibly succeed -- the quota resets once a
day, not once the suggested delay passes. So delay-presence alone can't
distinguish the two; the response's `quotaId` can (it contains "PerDay" for
this case), and that case is raised immediately as ``DailyQuotaExceededError``
instead of burning through max_retries and stalling the UI for minutes.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import TypeVar

from google.genai import errors

T = TypeVar("T")

_DELAY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s"),
    re.compile(r"retry in (\d+(?:\.\d+)?)s"),
)
_DAILY_QUOTA_RE = re.compile(r"PerDay", re.IGNORECASE)


class DailyQuotaExceededError(RuntimeError):
    """The Gemini free-tier per-day request quota is exhausted for this model."""


def _suggested_delay(exc: Exception) -> float | None:
    """Pull the server-suggested retry delay (seconds) out of the error, if any."""
    text = str(exc)
    for pattern in _DELAY_PATTERNS:
        match = pattern.search(text)
        if match:
            return float(match.group(1))
    return None


def _is_daily_quota_exhausted(exc: Exception) -> bool:
    return bool(_DAILY_QUOTA_RE.search(str(exc)))


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 6,
    base_delay: float = 2.0,
    on_retry: Callable[[int, int, float], None] | None = None,
) -> T:
    """Call ``fn`` and retry on rate-limit (429) or server (5xx) errors.

    ``on_retry(attempt, max_retries, wait_seconds)`` is invoked just before each
    sleep, so a caller with a UI (e.g. a Gradio progress bar) can show that a
    retry is in progress rather than looking stuck for however long the wait is.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            if code == 429 and _is_daily_quota_exhausted(exc):
                raise DailyQuotaExceededError(
                    "Gemini free-tier daily request quota is exhausted for this "
                    "model. It resets on a rolling 24h basis -- wait for the "
                    "reset or switch to a paid API tier for uninterrupted use."
                ) from exc
            delay = _suggested_delay(exc)
            retryable = (code == 429 and delay is not None) or (
                isinstance(code, int) and code >= 500
            )
            if not retryable or attempt == max_retries - 1:
                raise
            wait_seconds = (delay or base_delay * (2**attempt)) + 1.0
            if on_retry is not None:
                on_retry(attempt, max_retries, wait_seconds)
            time.sleep(wait_seconds)  # small buffer so the quota window resets
    raise RuntimeError("unreachable")  # pragma: no cover
