from __future__ import annotations

import ipaddress
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

# --- SSRF hardening ----------------------------------------------------------
# Media URLs embedded here come from user-authored posts, so a crafted URL can
# make this worker fetch internal targets (cloud metadata at 169.254.169.254,
# loopback/co-located services, ...). We resolve the host and refuse to fetch
# anything that lands on a never-legitimate range. Private/tailnet ranges stay
# reachable by default because media is served from the tailnet; operators who
# never do that can opt into stricter blocking with RECOMMENDATION_MEDIA_BLOCK_PRIVATE.

# Never a legitimate media origin — classic SSRF targets.
_ALWAYS_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "255.255.255.255/32",
        "::/128",
        "::1/128",
        "fe80::/10",
        "ff00::/8",
    )
)

# RFC1918 / ULA / CGNAT — valid for tailnet or docker-internal media hosts, so
# only rejected when the operator opts in.
_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",
        "fc00::/7",
    )
)

_DEFAULT_MAX_MEDIA_BYTES = 25 * 1024 * 1024


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _max_media_bytes() -> int:
    raw = os.getenv("RECOMMENDATION_MEDIA_MAX_BYTES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_MEDIA_BYTES
    return value if value > 0 else _DEFAULT_MAX_MEDIA_BYTES


def _allowed_media_hosts() -> frozenset[str]:
    raw = os.getenv("RECOMMENDATION_MEDIA_ALLOWED_HOSTS", "")
    return frozenset(host.strip().lower() for host in raw.split(",") if host.strip())


def _host_matches_allowlist(host: str, allowed: frozenset[str]) -> bool:
    host = host.lower()
    return any(host == entry or host.endswith("." + entry) for entry in allowed)


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if any(ip in network for network in _ALWAYS_BLOCKED_NETWORKS):
        return True
    if _env_flag("RECOMMENDATION_MEDIA_BLOCK_PRIVATE") and any(
        ip in network for network in _PRIVATE_NETWORKS
    ):
        return True
    return False


def _is_safe_media_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False

    allowed = _allowed_media_hosts()
    if allowed and not _host_matches_allowlist(parsed.hostname, allowed):
        return False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError):
        return False
    if not infos:
        return False

    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if _ip_is_blocked(ip):
            return False
    return True


def _read_capped(response: "requests.Response", max_bytes: int) -> bytes | None:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


@lru_cache(maxsize=1)
def _models():
    from sentence_transformers import SentenceTransformer

    return (
        SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1"),
        SentenceTransformer("clip-ViT-B-32"),
    )


def download_image(url: str) -> Any | None:
    if not _is_safe_media_url(url):
        return None

    from PIL import Image

    try:
        with requests.get(
            url, timeout=10, stream=True, allow_redirects=False
        ) as response:
            response.raise_for_status()
            content = _read_capped(response, _max_media_bytes())
        if not content:
            return None
        return Image.open(BytesIO(content)).convert("RGB")
    except Exception:
        return None


def extract_video_frames(url: str, interval_seconds: float = 10.0) -> list[Any]:
    if not _is_safe_media_url(url):
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
