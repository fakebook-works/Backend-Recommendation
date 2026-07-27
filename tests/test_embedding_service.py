import numpy as np
import pytest
import os
import sys
from types import SimpleNamespace

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


def test_ssrf_guard_blocks_private_media_hosts_by_default():
    assert embedding_service._is_safe_media_url("http://10.0.0.5/x.jpg") is False


def test_ssrf_guard_allows_only_explicitly_allowlisted_private_media_host(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_MEDIA_ALLOWED_HOSTS", "10.0.0.5")
    assert embedding_service._is_safe_media_url("http://10.0.0.5/x.jpg") is True
    assert embedding_service._is_safe_media_url("http://10.0.0.6/x.jpg") is False


def test_ssrf_guard_enforces_host_allowlist(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_MEDIA_ALLOWED_HOSTS", "cdn.example.com")
    # A public IP that is not on the allowlist is refused even though it resolves fine.
    assert embedding_service._is_safe_media_url("http://8.8.8.8/x.jpg") is False


def test_ssrf_guard_require_allowlist_rejects_otherwise_public_host(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_MEDIA_REQUIRE_ALLOWLIST", "true")
    assert embedding_service._is_safe_media_url("http://8.8.8.8/x.jpg") is False


def test_ssrf_guard_rejects_userinfo_fragment_and_invalid_port():
    assert embedding_service._is_safe_media_url("http://user@8.8.8.8/x.jpg") is False
    assert embedding_service._is_safe_media_url("http://8.8.8.8/x.jpg#fragment") is False
    assert embedding_service._is_safe_media_url("http://8.8.8.8:bad/x.jpg") is False


def test_relative_upload_url_resolves_only_to_configured_media_file_base(monkeypatch):
    monkeypatch.setenv("RECOMMENDATION_MEDIA_BASE_URL", "http://upload:4001")
    assert embedding_service._normalize_media_url("/media/files/a.jpg") == (
        "http://upload:4001/media/files/a.jpg"
    )
    assert embedding_service._normalize_media_url("/internal/media/delete") is None
    assert embedding_service._normalize_media_url("//169.254.169.254/latest/meta-data") is None


def test_read_capped_stops_over_limit():
    class _Streaming:
        headers = {}

        def iter_content(self, _chunk_size):
            yield b"a" * 10
            yield b"b" * 10

    assert embedding_service._read_capped(_Streaming(), max_bytes=15) is None
    assert embedding_service._read_capped(_Streaming(), max_bytes=100) == b"a" * 10 + b"b" * 10


class _Response:
    def __init__(self, chunks, status_code=200, headers=None):
        self._chunks = chunks
        self.status_code = status_code
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, _chunk_size):
        yield from self._chunks


def test_capped_download_refuses_redirect_and_does_not_follow(monkeypatch):
    captured = {}

    def fake_get(_url, **kwargs):
        captured.update(kwargs)
        return _Response([b"redirect"], status_code=302)

    monkeypatch.setattr(embedding_service, "resilient_get", fake_get)
    assert embedding_service._download_capped_to_temp("http://8.8.8.8/a.mp4", 100) is None
    assert captured["allow_redirects"] is False


def test_capped_download_checks_streamed_size_not_only_content_length(monkeypatch):
    monkeypatch.setattr(
        embedding_service,
        "resilient_get",
        lambda *_args, **_kwargs: _Response([b"a" * 8, b"b" * 8], headers={"Content-Length": "8"}),
    )
    assert embedding_service._download_capped_to_temp("http://8.8.8.8/a.mp4", 12) is None


def test_video_decoder_receives_local_bounded_file_and_temp_is_deleted(monkeypatch):
    seen = {}

    class FakeCapture:
        def __init__(self, path):
            seen["path"] = path
            seen["exists_during_open"] = os.path.exists(path)

        def isOpened(self):
            return False

        def release(self):
            seen["released"] = True

    monkeypatch.setattr(
        embedding_service,
        "resilient_get",
        lambda *_args, **_kwargs: _Response([b"bounded-video"]),
    )
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(VideoCapture=FakeCapture))
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))

    assert embedding_service.extract_video_frames("http://8.8.8.8/a.mp4") == []
    assert seen["exists_during_open"] is True
    assert seen["released"] is True
    assert not os.path.exists(seen["path"])
