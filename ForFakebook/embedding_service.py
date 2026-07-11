import requests
from io import BytesIO
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import cv2

# Initialize Multilingual CLIP model for text (512 dimensions)
model = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

# Initialize original CLIP model for images/videos (512 dimensions)
image_model = SentenceTransformer("clip-ViT-B-32")

def download_image(url: str) -> Image.Image:
    """Download image from URL and convert to PIL Image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

def extract_video_frames(url: str, interval_seconds: float = 10.0) -> list[Image.Image]:
    """Stream video from URL and extract frames at regular intervals (default: every 10s)"""
    frames = []
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Warning: Could not open video {url}")
            return frames
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Fallback if FPS is not read correctly
        if fps <= 0:
            fps = 25.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(interval_seconds * fps)
        if frame_step <= 0:
            frame_step = 1
            
        for frame_idx in range(0, total_frames, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV format) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            frames.append(pil_img)
            
        cap.release()
    except Exception as e:
        print(f"Error extracting frames from video {url}: {e}")
    return frames

def download_media(url: str) -> list[Image.Image]:
    """Download media from URL (image or video) and convert to list of PIL Images.
    If it's an image, returns a list containing 1 PIL Image.
    If it's a video, extracts frames every 10 seconds and returns a list of PIL Images.
    """
    is_video = False
    lower_url = url.lower()
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".3gp", ".ogg")
    if any(ext in lower_url for ext in video_extensions):
        is_video = True
    else:
        # Check Content-Type via a quick HEAD request
        try:
            head_res = requests.head(url, timeout=5, allow_redirects=True)
            content_type = head_res.headers.get("Content-Type", "")
            if "video" in content_type:
                is_video = True
        except Exception:
            pass

    if is_video:
        return extract_video_frames(url, interval_seconds=10.0)
    else:
        # Treat as image
        img = download_image(url)
        return [img] if img else []

def generate_multimodal_embedding(title: str, image_urls: list = None):
    # 1. Generate text embedding
    text_emb = model.encode(title, normalize_embeddings=True)
    
    # 2. Generate image / video frame embeddings
    image_embs = []
    if image_urls:
        # Download and process media concurrently
        with ThreadPoolExecutor(max_workers=min(len(image_urls), 8)) as executor:
            results = list(executor.map(download_media, image_urls))
        
        # Flatten the list of lists of images
        valid_images = []
        for img_list in results:
            for img in img_list:
                if img is not None:
                    valid_images.append(img)
        
        if valid_images:
            # Batch encode all valid images/frames
            image_embs = image_model.encode(valid_images, normalize_embeddings=True)
                
    # 3. Fuse Text & Image embeddings
    if len(image_embs) > 0:
        mean_image_emb = np.mean(image_embs, axis=0)
        # Combine: 60% text, 40% image/video mean
        combined_emb = 0.6 * text_emb + 0.4 * mean_image_emb
        # Normalize to unit vector
        combined_emb = combined_emb / np.linalg.norm(combined_emb)
        return combined_emb.tolist()
    
    return text_emb.tolist()
