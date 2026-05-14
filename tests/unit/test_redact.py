"""Tests for audit log query redaction."""
import pytest
from src.audit.redact import redact_query, _MAX_QUERY_LENGTH


def test_none_returns_none():
    assert redact_query(None) is None


def test_clean_query_unchanged():
    q = "How does hybrid retrieval work?"
    assert redact_query(q) == q


def test_email_redacted():
    result = redact_query("Contact admin@example.com for help")
    assert "[EMAIL]" in result
    assert "admin@example.com" not in result


def test_bearer_token_redacted():
    result = redact_query("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc123")
    assert "[TOKEN]" in result
    assert "eyJ" not in result


def test_api_key_sk_redacted():
    result = redact_query("Use sk-abcdefghijklmnop to access the API")
    assert "[API_KEY]" in result
    assert "sk-abcdefghijklmnop" not in result


def test_api_key_pk_redacted():
    result = redact_query("pk-lf-supersecretkey12345")
    assert "[API_KEY]" in result


def test_hex_secret_redacted():
    secret = "a" * 32
    result = redact_query(f"token is {secret}")
    assert "[SECRET]" in result
    assert secret not in result


def test_truncation_at_max_length():
    long_query = "x" * (_MAX_QUERY_LENGTH + 100)
    result = redact_query(long_query)
    assert "[truncated]" in result
    assert len(result) < len(long_query)


def test_multiple_patterns_in_one_query():
    q = "email admin@corp.com key sk-abc12345678 token Bearer eyABC123"
    result = redact_query(q)
    assert "admin@corp.com" not in result
    assert "sk-abc12345678" not in result
