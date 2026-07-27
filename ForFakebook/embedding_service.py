from __future__ import annotations

import ipaddress
import os
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

from .http_resilience import resilient_get

# --- SSRF hardening ----------------------------------------------------------
# Media URLs embedded here come from user-authored posts, so a crafted URL can
# make this worker fetch internal targets (cloud metadata at 169.254.169.254,
# loopback/co-located services, ...). We resolve the host and refuse to fetch
# anything that lands on a non-public range. Private/tailnet ranges are reachable
# only for an exact deployment allowlist entry (for example the Docker `upload`
# service), so user-provided URLs cannot turn this worker into an internal proxy.

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
_DEFAULT_MAX_VIDEO_BYTES = 100 * 1024 * 1024
_DEFAULT_MAX_VIDEO_FRAMES = 24
_DEFAULT_MAX_MEDIA_PIXELS = 40_000_000


def _env_flag(name: str) -> bool:
    normalized = os.getenv(name, "").strip().lower()
    if normalized in {"", "0", "false", "no", "off"}:
        return False
    # Invalid security-flag values fail closed by enabling the restriction.
    return True


def _max_media_bytes() -> int:
    raw = os.getenv("RECOMMENDATION_MEDIA_MAX_BYTES", "").strip()
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_MEDIA_BYTES
    return value if value > 0 else _DEFAULT_MAX_MEDIA_BYTES


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _max_video_bytes() -> int:
    return _positive_env_int("RECOMMENDATION_VIDEO_MAX_BYTES", _DEFAULT_MAX_VIDEO_BYTES)


def _max_video_frames() -> int:
    return _positive_env_int("RECOMMENDATION_MAX_VIDEO_FRAMES", _DEFAULT_MAX_VIDEO_FRAMES)


def _max_media_pixels() -> int:
    return _positive_env_int("RECOMMENDATION_MAX_MEDIA_PIXELS", _DEFAULT_MAX_MEDIA_PIXELS)


def _allowed_media_hosts() -> frozenset[str]:
    raw = os.getenv("RECOMMENDATION_MEDIA_ALLOWED_HOSTS", "")
    return frozenset(
        host.strip().lower().rstrip(".") for host in raw.split(",") if host.strip()
    )


def _normalize_media_url(url: str) -> str | None:
    value = url.strip()
    parsed = urlparse(value)
    if parsed.scheme:
        return value
    if parsed.netloc:
        return None

    # Upload Server deliberately returns same-origin relative URLs. Resolve only
    # its generated-file route against a deployment-controlled base; never turn
    # an arbitrary relative user value into an internal-service request.
    if not value.startswith("/media/files/"):
        return None
    base_url = os.getenv("RECOMMENDATION_MEDIA_BASE_URL", "").strip().rstrip("/")
    parsed_base = urlparse(base_url)
    if parsed_base.scheme not in {"http", "https"} or not parsed_base.hostname:
        return None
    return base_url + value


def _host_matches_allowlist(host: str, allowed: frozenset[str]) -> bool:
    host = host.lower().rstrip(".")
    return host in allowed


def _ip_is_blocked(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_private: bool = False,
) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_unspecified
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or any(ip in network for network in _ALWAYS_BLOCKED_NETWORKS)
    ):
        return True
    if not allow_private and (
        ip.is_private or any(ip in network for network in _PRIVATE_NETWORKS)
    ):
        return True
    return False


def _is_safe_media_url(url: str) -> bool:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        return False

    allowed = _allowed_media_hosts()
    is_allowlisted = bool(allowed and _host_matches_allowlist(parsed.hostname, allowed))
    if allowed and not is_allowlisted:
        return False
    if _env_flag("RECOMMENDATION_MEDIA_REQUIRE_ALLOWLIST") and not is_allowlisted:
        return False

    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return False
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
        # Private/Tailscale media origins are accepted only when their hostname is
        # explicitly controlled by the deployment allowlist. Link-local, loopback,
        # multicast and reserved ranges are never bypassed by the allowlist.
        if _ip_is_blocked(ip, allow_private=is_allowlisted):
            return False
    return True


def _read_capped(response: "requests.Response", max_bytes: int) -> bytes | None:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            if int(content_length) > max_bytes:
                return None
        except ValueError:
            return None

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


def _download_capped_to_temp(url: str, max_bytes: int) -> str | None:
    """Download an already-validated URL without redirects into a bounded temp file."""
    suffix = os.path.splitext(urlparse(url).path)[1][:10]
    temporary_path: str | None = None
    try:
        with resilient_get(
            url,
            timeout=(3.05, 15),
            stream=True,
            allow_redirects=False,
        ) as response:
            status_code = getattr(response, "status_code", 200)
            if status_code < 200 or status_code >= 300:
                return None
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > max_bytes:
                        return None
                except ValueError:
                    return None

            total = 0
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary_path = temporary.name
                for chunk in response.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        temporary.close()
                        os.unlink(temporary_path)
                        return None
                    temporary.write(chunk)
        if temporary_path and os.path.getsize(temporary_path) > 0:
            return temporary_path
        if temporary_path:
            os.unlink(temporary_path)
        return None
    except Exception:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        return None


@lru_cache(maxsize=1)
def _models():
    from sentence_transformers import SentenceTransformer

    return (
        SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1"),
        SentenceTransformer("clip-ViT-B-32"),
    )


def download_image(url: str) -> Any | None:
    url = _normalize_media_url(url) or ""
    if not _is_safe_media_url(url):
        return None

    from PIL import Image

    try:
        with resilient_get(
            url, timeout=(3.05, 10), stream=True, allow_redirects=False
        ) as response:
            status_code = getattr(response, "status_code", 200)
            if status_code < 200 or status_code >= 300:
                return None
            response.raise_for_status()
            content = _read_capped(response, _max_media_bytes())
        if not content:
            return None
        with Image.open(BytesIO(content)) as image:
            if image.width <= 0 or image.height <= 0 or image.width * image.height > _max_media_pixels():
                return None
            image.load()
            return image.convert("RGB")
    except Exception:
        return None


def extract_video_frames(url: str, interval_seconds: float = 10.0) -> list[Any]:
    url = _normalize_media_url(url) or ""
    if not _is_safe_media_url(url):
        return []

    import cv2
    from PIL import Image

    frames: list[Any] = []
    local_path = _download_capped_to_temp(url, _max_video_bytes())
    if not local_path:
        return frames

    capture = cv2.VideoCapture(local_path)
    try:
        if not capture.isOpened():
            return frames

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0 or width * height > _max_media_pixels():
            return frames

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = max(1, int(interval_seconds * fps))

        for frame_index in range(0, total_frames, frame_step):
            if len(frames) >= _max_video_frames():
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
        try:
            os.unlink(local_path)
        except OSError:
            pass

    return frames


def download_media(url: str) -> list[Any]:
    normalized_url = _normalize_media_url(url)
    if not normalized_url:
        return []
    lower_url = urlparse(normalized_url).path.lower()
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp", ".ogg")
    if any(extension in lower_url for extension in video_extensions):
        return extract_video_frames(normalized_url)

    image = download_image(normalized_url)
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
