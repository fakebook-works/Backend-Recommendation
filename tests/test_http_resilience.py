from ForFakebook import http_resilience


def test_transport_retries_only_safe_methods():
    assert http_resilience._retry.allowed_methods == frozenset({"GET", "HEAD", "OPTIONS"})
    assert "POST" not in http_resilience._retry.allowed_methods
    assert "DELETE" not in http_resilience._retry.allowed_methods
