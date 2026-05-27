"""Redaction helpers for provider diagnostics and errors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SECRET_QUERY_KEYS = {"apikey", "api_key", "key", "token", "access_token"}
_SECRET_HEADER_KEYS = {"authorization", "x-api-key", "api-key"}
_QUERY_SECRET_RE = re.compile(
    r"(?i)([?&](?:apiKey|api_key|key|token|access_token)=)([^&#\s'\"]+)"
)
_HEADER_SECRET_RE = re.compile(
    r"(?i)\b(authorization|x-api-key|api-key)\s*[:=]\s*([^\s,;'\"]+)"
)


def redact_url(url: Any) -> str:
    """Return ``url`` with known secret query params replaced."""
    text = str(url or "")
    if not text:
        return text
    try:
        parts = urlsplit(text)
    except ValueError:
        return sanitize_text(text)
    if not parts.query:
        return text
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        query.append((key, "[redacted]" if key.lower() in _SECRET_QUERY_KEYS else value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def redact_headers(headers: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a copy of headers with auth values removed."""
    out: dict[str, Any] = {}
    for key, value in (headers or {}).items():
        out[key] = "[redacted]" if str(key).lower() in _SECRET_HEADER_KEYS else value
    return out


def sanitize_text(value: Any, *, limit: int | None = None) -> str:
    """Redact secret tokens inside arbitrary exception/log text."""
    text = str(value or "")
    text = _QUERY_SECRET_RE.sub(r"\1[redacted]", text)
    text = _HEADER_SECRET_RE.sub(r"\1: [redacted]", text)
    if limit is not None and len(text) > limit:
        return text[:limit]
    return text

