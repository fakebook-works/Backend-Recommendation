"""Conservative outbound HTTP resilience for safe methods only."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


_retry = Retry(
    total=2,
    connect=2,
    read=2,
    status=2,
    backoff_factor=0.2,
    status_forcelist=(429, 502, 503, 504),
    allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
    respect_retry_after_header=True,
    raise_on_status=False,
)
_session = requests.Session()
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=20, pool_maxsize=50)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def resilient_get(url: str, **kwargs):
    return _session.get(url, **kwargs)
