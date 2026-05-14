"""Query redaction — removes sensitive patterns before audit log persistence."""
import re

_MAX_QUERY_LENGTH = 500

_PATTERNS = [
    # Email addresses
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    # Phone numbers (E.164, US, international)
    (re.compile(r"(?<!\d)(\+?1?\s?)?(\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]\d{4}(?!\d)"), "[PHONE]"),
    # Bearer tokens / Authorization headers
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*"), "[TOKEN]"),
    # API keys — common prefixes (sk-, pk-, lf-, ghp_, etc.)
    (re.compile(r"(?i)(sk|pk|lf|ghp|gho|ghu|ghs|ghr|xoxb|xoxp|xoxr|xoxa)[-_][A-Za-z0-9]{8,}"), "[API_KEY]"),
    # High-entropy strings likely to be secrets (32+ hex chars)
    (re.compile(r"(?<![a-fA-F0-9])[a-fA-F0-9]{32,}(?![a-fA-F0-9])"), "[SECRET]"),
]


def redact_query(query: str | None) -> str | None:
    """Apply pattern-based redaction and truncate to safe length.

    Returns None if input is None.
    """
    if query is None:
        return None

    result = query
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)

    if len(result) > _MAX_QUERY_LENGTH:
        result = result[:_MAX_QUERY_LENGTH] + "…[truncated]"

    return result
