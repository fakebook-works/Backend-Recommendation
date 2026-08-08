from __future__ import annotations

import unicodedata
from urllib.parse import urlsplit


MAX_CONTENT_LENGTH = 63_206
MAX_MEDIA_URL_LENGTH = 2_048
MAX_MEDIA_COUNT = 10
MAX_COMBINING_MARKS = 256


def normalize_text(value: str, maximum: int, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    if len(value) > maximum * 2 + 32:
        raise ValueError(f"text must not exceed {maximum} Unicode characters")
    normalized = unicodedata.normalize("NFKC", value.replace("\r\n", "\n").replace("\r", "\n"))
    result: list[str] = []
    combining = 0
    consecutive = 0
    for character in normalized:
        category = unicodedata.category(character)
        # Preserve ordinary LF/TAB in post text while rejecting every other
        # control/formatting code point. This keeps multiline content usable
        # without allowing terminal/bidi/private-use abuse.
        if character in "\n\t":
            result.append(character)
            consecutive = 0
            continue
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            raise ValueError("text contains an unsupported control or formatting character")
        if category in {"Zl", "Zp"}:
            raise ValueError("text contains an unsupported line separator")
        if category in {"Mn", "Mc", "Me"}:
            combining += 1
            consecutive += 1
            if consecutive > 3 or combining > min(MAX_COMBINING_MARKS, max(16, maximum // 8)):
                raise ValueError("text contains excessive combining marks")
        else:
            consecutive = 0
        result.append(character)
    normalized = "".join(result)
    if len(normalized) > maximum:
        raise ValueError(f"text must not exceed {maximum} Unicode characters")
    if not allow_empty and not normalized.strip():
        raise ValueError("text must contain at least one non-whitespace character")
    return normalized


def normalize_media_urls(values: list[str]) -> list[str]:
    if len(values) > MAX_MEDIA_COUNT:
        raise ValueError(f"mediaUrls must contain at most {MAX_MEDIA_COUNT} items")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or len(value) > MAX_MEDIA_URL_LENGTH:
            raise ValueError("mediaUrls contains an invalid URL")
        # Media locations are generated managed paths or HTTPS protocol metadata, not
        # display prose. Keeping this boundary printable ASCII prevents bidi/private-use
        # and combining-heavy URL strings from reaching logs, storage or remote fetches;
        # international hosts must use their standard IDNA/punycode representation.
        if not value.isascii() or any(not "!" <= character <= "~" for character in value):
            raise ValueError("mediaUrls contains an invalid URL")
        parsed = urlsplit(value)
        if parsed.scheme and parsed.scheme.lower() not in {"http", "https"}:
            raise ValueError("mediaUrls contains an invalid URL scheme")
        if value.startswith("//"):
            raise ValueError("mediaUrls contains an invalid URL")
        key = value.casefold()
        if key in seen:
            raise ValueError("mediaUrls contains duplicate URLs")
        seen.add(key)
        result.append(value)
    return result


def validate_idempotency_key(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.isascii():
        raise ValueError("Idempotency-Key must contain 1 to 128 ASCII characters")
    if any(character.isspace() or ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValueError("Idempotency-Key contains an invalid character")
    return value
