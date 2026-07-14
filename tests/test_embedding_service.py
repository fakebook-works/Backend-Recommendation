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
