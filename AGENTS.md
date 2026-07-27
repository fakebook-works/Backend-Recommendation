# Recommendation service agent rules

When embedded in the Fakebook workspace, also read the root API security contract.

- Internal endpoints require HMAC signatures and Redis nonce replay protection; Redis
  outage fails closed.
- Never fetch a user-controlled URL directly. Use the existing media URL normalizer,
  exact host allowlist, DNS/IP guard, dangerous-range blocking, redirect disablement,
  byte/time cap and bounded temporary-file decoder.
- Bound candidate count, media pixels/frames, worker concurrency and database queries.
- Automatic retry is allowed only for GET/HEAD/OPTIONS.
- Runtime DB access uses the recommendation role; public schema USAGE exists only for pgvector.
- Do not put content, media bytes, tokens or signed headers into traces/logs.
- Pin dependencies and run pip-audit after changing requirements.

Run python -m pytest -q and add SSRF/timeout/boundary/replay tests for API changes.
