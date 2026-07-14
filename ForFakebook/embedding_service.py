from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests


@lru_cache(maxsize=1)
def _models():
    from sentence_transformers import SentenceTransformer

    return (
        SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1"),
        SentenceTransformer("clip-ViT-B-32"),
    )


def _is_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_image(url: str) -> Any | None:
    if not _is_http_url(url):
        return None

    from PIL import Image

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        return None


def extract_video_frames(url: str, interval_seconds: float = 10.0) -> list[Any]:
    if not _is_http_url(url):
        return []

    import cv2
    from PIL import Image

    frames: list[Any] = []
    capture = cv2.VideoCapture(url)
    try:
        if not capture.isOpened():
            return frames

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(interval_seconds * fps))

        for frame_index in range(0, total_frames, frame_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()

    return frames


def download_media(url: str) -> list[Any]:
    lower_url = url.lower()
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp", ".ogg")
    if any(extension in lower_url for extension in video_extensions):
        return extract_video_frames(url)

    image = download_image(url)
    return [image] if image is not None else []


def generate_multimodal_embedding(content: str, media_urls: list[str] | None = None) -> list[float]:
    text_model, image_model = _models()
    text_embedding = None
    if content.strip():
        text_embedding = np.asarray(
            text_model.encode(content, normalize_embeddings=True),
            dtype=float,
        )

    valid_images: list[Any] = []
    urls = [url.strip() for url in media_urls or [] if url.strip()]
    if urls:
        with ThreadPoolExecutor(max_workers=min(len(urls), 8)) as executor:
            for images in executor.map(download_media, urls):
                valid_images.extend(image for image in images if image is not None)

    if not valid_images:
        if text_embedding is None:
            raise ValueError("content is empty and no media could be processed")
        return text_embedding.tolist()

    image_embeddings = image_model.encode(valid_images, normalize_embeddings=True)
    media_embedding = np.mean(np.asarray(image_embeddings, dtype=float), axis=0)
    if text_embedding is None:
        norm = np.linalg.norm(media_embedding)
        if norm == 0:
            raise ValueError("media embedding has zero magnitude")
        return (media_embedding / norm).tolist()

    combined = 0.6 * text_embedding + 0.4 * media_embedding
    combined = combined / np.linalg.norm(combined)
    return combined.tolist()
