"""Tests for the API retry helper. No network or API key required."""

import pytest
from google.genai import errors

import credit_risk_analyzer.retry as retry_mod


def _api_error(code: int, message: str | None = None):
    return errors.APIError(
        code,
        {"error": {"message": message or f"status {code}", "status": "RESOURCE_EXHAUSTED"}},
    )


def test_retries_then_succeeds(monkeypatch):
    """A transient per-minute rate limit (has a server-suggested retryDelay) is retried."""
    monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)  # don't actually wait
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _api_error(429, "Resource exhausted, retry in 1s")
        return "ok"

    assert retry_mod.call_with_retry(fn, max_retries=3) == "ok"
    assert calls["n"] == 2


def test_quota_exhaustion_fails_fast(monkeypatch):
    """A hard quota exhaustion (no retryDelay) is raised immediately, not retried."""
    monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _api_error(
            429, "You exceeded your current quota, please check your plan and billing details."
        )

    with pytest.raises(errors.APIError):
        retry_mod.call_with_retry(fn, max_retries=6)
    assert calls["n"] == 1  # no retries wasted on an unrecoverable-in-the-short-term error


def test_daily_quota_fails_fast_even_with_retry_delay(monkeypatch):
    """A per-day quota error still carries a retryDelay (observed live), so delay
    presence alone can't be used to retry it -- quotaId containing "PerDay"
    must fail fast instead, since the quota won't reset within that delay."""
    monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _api_error(
            429,
            "Quota exceeded for metric: generativelanguage.googleapis.com/"
            "generate_content_free_tier_requests, limit: 20, model: "
            "gemini-2.5-flash. quotaId: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier. "
            "Please retry in 58.7s.",
        )

    with pytest.raises(retry_mod.DailyQuotaExceededError):
        retry_mod.call_with_retry(fn, max_retries=6)
    assert calls["n"] == 1


def test_non_retryable_propagates(monkeypatch):
    monkeypatch.setattr(retry_mod.time, "sleep", lambda _s: None)

    def fn():
        raise _api_error(400)  # client error, not a rate limit

    with pytest.raises(errors.APIError):
        retry_mod.call_with_retry(fn, max_retries=3)


def test_suggested_delay_parsed():
    exc = Exception("please retry in 12.5s")
    assert retry_mod._suggested_delay(exc) == 12.5
