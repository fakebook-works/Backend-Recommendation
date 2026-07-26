from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
import time
from urllib.parse import urlencode, urlsplit


TIMESTAMP_HEADER = "X-Internal-Timestamp"
NONCE_HEADER = "X-Internal-Nonce"
SIGNATURE_HEADER = "X-Internal-Signature"


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def build_canonical(
    method: str,
    path_and_query: str,
    timestamp: int,
    nonce: str,
    body: bytes,
) -> str:
    body_hash = hashlib.sha256(body or b"").hexdigest()
    return (
        f"v1\n{method.upper()}\n{path_and_query}\n"
        f"{timestamp}\n{nonce}\n{body_hash}"
    )


def sign(
    secret: str,
    method: str,
    path_and_query: str,
    timestamp: int,
    nonce: str,
    body: bytes,
) -> str:
    canonical = build_canonical(method, path_and_query, timestamp, nonce, body)
    return hmac.new(
        secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def path_and_query(url: str, params: dict | None = None) -> str:
    parsed = urlsplit(url)
    path = parsed.path or "/"
    query_parts = []
    if parsed.query:
        query_parts.append(parsed.query)
    if params:
        query_parts.append(urlencode(params, doseq=True))
    return path + (("?" + "&".join(query_parts)) if query_parts else "")


def signed_headers(
    secret: str,
    method: str,
    url: str,
    *,
    body: bytes = b"",
    params: dict | None = None,
    legacy_header: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    timestamp = int(time.time()) if timestamp is None else timestamp
    nonce = secrets.token_hex(16) if nonce is None else nonce
    headers = dict(extra_headers or {})
    headers[TIMESTAMP_HEADER] = str(timestamp)
    headers[NONCE_HEADER] = nonce
    headers[SIGNATURE_HEADER] = sign(
        secret,
        method,
        path_and_query(url, params),
        timestamp,
        nonce,
        body,
    )
    if legacy_header and env_flag("INTERNAL_AUTH_SEND_LEGACY_SECRET", default=True):
        headers[legacy_header] = secret
    return headers


class SignatureValidator:
    NO_SIGNATURE = "no_signature"
    VALID = "valid"
    INVALID = "invalid"

    def __init__(self):
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()
        self._validation_count = 0

    def validate(
        self,
        secret: str,
        method: str,
        request_path_and_query: str,
        body: bytes,
        timestamp_header: str | None,
        nonce_header: str | None,
        signature_header: str | None,
        *,
        now: float | None = None,
    ) -> str:
        any_present = any(
            value is not None
            for value in (timestamp_header, nonce_header, signature_header)
        )
        if not any_present:
            return self.NO_SIGNATURE
        if not timestamp_header or not nonce_header or not signature_header:
            return self.INVALID

        try:
            timestamp = int(timestamp_header)
        except (TypeError, ValueError):
            return self.INVALID

        if (
            len(nonce_header) != 32
            or any(character not in "0123456789abcdefABCDEF" for character in nonce_header)
            or len(signature_header) != 64
        ):
            return self.INVALID

        now = time.time() if now is None else now
        try:
            skew = int(os.getenv("INTERNAL_AUTH_CLOCK_SKEW_SECONDS", "300"))
            retention = int(os.getenv("INTERNAL_AUTH_NONCE_RETENTION_SECONDS", "900"))
        except ValueError:
            return self.INVALID
        if not (30 <= skew <= 900 and skew <= retention <= 3600):
            return self.INVALID
        if timestamp > now + skew or timestamp < now - skew:
            return self.INVALID

        expected = sign(
            secret,
            method,
            request_path_and_query,
            timestamp,
            nonce_header,
            body,
        )
        if not hmac.compare_digest(expected, signature_header.lower()):
            return self.INVALID

        expires_at = now + retention
        with self._lock:
            existing = self._seen.get(nonce_header)
            if existing is not None and existing >= now:
                return self.INVALID
            self._seen[nonce_header] = expires_at
            self._validation_count += 1
            if self._validation_count % 256 == 0:
                self._seen = {
                    nonce: expiry
                    for nonce, expiry in self._seen.items()
                    if expiry >= now
                }
        return self.VALID


validator = SignatureValidator()
