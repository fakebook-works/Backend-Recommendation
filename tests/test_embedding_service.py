import numpy as np
import pytest

from ForFakebook import embedding_service


class FakeTextModel:
    def __init__(self):
        self.calls = []

    def encode(self, content, normalize_embeddings):
        self.calls.append((content, normalize_embeddings))
        vector = np.zeros(512)
        vector[0] = 1.0
        return vector


class FakeImageModel:
    def encode(self, images, normalize_embeddings):
        vector = np.zeros(512)
        vector[0] = 3.0
        vector[1] = 4.0
        return np.asarray([vector for _ in images])


def test_media_only_post_uses_normalized_media_embedding(monkeypatch):
    text_model = FakeTextModel()
    image_model = FakeImageModel()
    monkeypatch.setattr(embedding_service, "_models", lambda: (text_model, image_model))
    monkeypatch.setattr(embedding_service, "download_media", lambda _url: [object()])

    result = np.asarray(
        embedding_service.generate_multimodal_embedding(
            "",
            ["https://example.com/media-without-extension"],
        )
    )

    assert text_model.calls == []
    assert result.shape == (512,)
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert np.isclose(result[0], 0.6)
    assert np.isclose(result[1], 0.8)


def test_media_only_post_fails_when_no_media_can_be_processed(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "_models",
        lambda: (FakeTextModel(), FakeImageModel()),
    )
    monkeypatch.setattr(embedding_service, "download_media", lambda _url: [])

    with pytest.raises(ValueError, match="no media could be processed"):
        embedding_service.generate_multimodal_embedding(
            "",
            ["https://example.com/unavailable.jpg"],
        )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://8.8.8.8/x.jpg",  # non-http scheme
        "http:///no-host.jpg",  # missing host
        "http://127.0.0.1/x.jpg",  # loopback
        "http://[::1]/x.jpg",  # loopback v6
        "http://0.0.0.0/x.jpg",  # unspecified
        "http://169.254.169.254/latest/meta-data",  # cloud metadata (SSRF crown jewel)
    ],
)
def test_ssrf_guard_rejects_dangerous_media_urls(url):
    # Literal IPs are validated without any DNS lookup, so this stays offline.
    assert embedding_service._is_safe_media_url(url) is False


def test_ssrf_guard_allows_public_literal_ip():
    assert embedding_service._is_safe_media_url("http://8.8.8.8/x.jpg") is True


def test_ssrf_guard_allows_private_media_hosts_by_default():
    assert embedding_service._is_safe_media_url("http://10.0.0.5/x.jpg") is True


def test_ssrf_guard_blocks_private_media_hosts_when_opted_in(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_MEDIA_BLOCK_PRIVATE", "1")
    assert embedding_service._is_safe_media_url("http://10.0.0.5/x.jpg") is False
    assert embedding_service._is_safe_media_url("http://192.168.1.10/x.jpg") is False


def test_ssrf_guard_enforces_host_allowlist(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_MEDIA_ALLOWED_HOSTS", "cdn.example.com")
    # A public IP that is not on the allowlist is refused even though it resolves fine.
    assert embedding_service._is_safe_media_url("http://8.8.8.8/x.jpg") is False


def test_read_capped_stops_over_limit():
    class _Streaming:
        def iter_content(self, _chunk_size):
            yield b"a" * 10
            yield b"b" * 10

    assert embedding_service._read_capped(_Streaming(), max_bytes=15) is None
    assert embedding_service._read_capped(_Streaming(), max_bytes=100) == b"a" * 10 + b"b" * 10
