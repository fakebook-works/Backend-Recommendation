import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from concurrent.futures import ThreadPoolExecutor
import strawberry
from strawberry.fastapi import GraphQLRouter
import cv2

# Setup Database connection
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:postgres@localhost:5432/fakebook"
)

try:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Warning: Database engine connection failed: {e}")
    SessionLocal = None

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
    # Detect video by file extension or content-type
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
            # download_media returns a list of PIL Images for each URL
            results = list(executor.map(download_media, image_urls))
        
        # Flatten the list of lists of images
        valid_images = []
        for img_list in results:
            for img in img_list:
                if img is not None:
                    valid_images.append(img)
        
        if valid_images:
            # Batch encode all valid images/frames at once for maximum performance using the image model
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


def save_embedding(db, post_id: int, embedding: list):
    db.execute(
        text("""
            INSERT INTO post_embeddings (post_id, embedding)
            VALUES (:post_id, :embedding)
            ON CONFLICT (post_id) DO UPDATE 
            SET embedding = EXCLUDED.embedding, updated_at = NOW();
        """),
        {
            "post_id": post_id,
            "embedding": embedding,
        }
    )
    db.commit()

def save_user_embedding(db, user_id: int, embedding: list):
    db.execute(
        text("""
            INSERT INTO user_embeddings (user_id, embedding, updated_at)
            VALUES (:user_id, :embedding, NOW())
            ON CONFLICT (user_id) DO UPDATE 
            SET embedding = EXCLUDED.embedding, updated_at = NOW();
        """),
        {
            "user_id": user_id,
            "embedding": embedding,
        }
    )
    db.commit()

def parse_vector(val):
    """Safely parse vector returned from database to a numpy array"""
    if isinstance(val, str):
        # Format usually is "[0.1, 0.2, ...]"
        clean_val = val.strip("[]")
        if not clean_val:
            return np.zeros(512)
        return np.array([float(x) for x in clean_val.split(",")])
    elif isinstance(val, list):
        return np.array(val)
    elif isinstance(val, np.ndarray):
        return val
    return np.array(val)

app = FastAPI()

# Enable CORS so the React frontend can call it directly from the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@strawberry.type
class EmbeddingResponse:
    success: bool
    post_id: int

@strawberry.type
class UserEmbeddingResponse:
    success: bool
    user_id: int
    message: str | None = None

@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Hello from Embedding GraphQL API"

@strawberry.type
class Mutation:
    @strawberry.mutation
    def initialize_user_embedding(self, user_id: int) -> UserEmbeddingResponse:
        if SessionLocal is None:
            raise Exception("Database connection is not initialized.")
            
        # Generate random 512-dim vector
        rand_emb = np.random.randn(512)
        # Normalize
        rand_emb = rand_emb / np.linalg.norm(rand_emb)
        
        db = SessionLocal()
        try:
            save_user_embedding(db, user_id, rand_emb.tolist())
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to initialize user embedding: {str(e)}")
        finally:
            db.close()
            
        return UserEmbeddingResponse(success=True, user_id=user_id, message="Random embedding initialized")

    @strawberry.mutation
    def create_post_embedding(
        self, 
        post_id: int, 
        title: str, 
        image_urls: list[str] | None = None
    ) -> EmbeddingResponse:
        if SessionLocal is None:
            raise Exception("Database connection is not initialized.")
            
        urls = [str(url).strip() for url in image_urls] if image_urls else []
        embedding = generate_multimodal_embedding(title, urls)
        
        db = SessionLocal()
        try:
            save_embedding(db, post_id, embedding)
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to save post embedding: {str(e)}")
        finally:
            db.close()
            
        return EmbeddingResponse(success=True, post_id=post_id)

    @strawberry.mutation
    def update_user_embedding(
        self, 
        user_id: int, 
        post_id: int, 
        view_time: float
    ) -> UserEmbeddingResponse:
        if SessionLocal is None:
            raise Exception("Database connection is not initialized.")
            
        db = SessionLocal()
        try:
            # 1. Get post embedding
            post_row = db.execute(
                text("SELECT embedding FROM post_embeddings WHERE post_id = :post_id"),
                {"post_id": post_id}
            ).fetchone()
            
            if not post_row:
                raise Exception(f"Post embedding for post_id {post_id} not found.")
                
            post_emb = parse_vector(post_row[0])
            
            # 2. Get user embedding
            user_row = db.execute(
                text("SELECT embedding FROM user_embeddings WHERE user_id = :user_id"),
                {"user_id": user_id}
            ).fetchone()
            
            if user_row:
                user_emb = parse_vector(user_row[0])
                # Calculate weight from view_time
                w = min(view_time / 10.0, 2.0)
                # Incremental weighted update
                new_emb = user_emb + w * post_emb
                # Normalize
                new_emb = new_emb / np.linalg.norm(new_emb)
            else:
                # Initialize user with the post embedding directly
                new_emb = post_emb
                
            # 3. Save
            save_user_embedding(db, user_id, new_emb.tolist())
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Failed to update user embedding: {str(e)}")
        finally:
            db.close()
            
        return UserEmbeddingResponse(
            success=True, 
            user_id=user_id, 
            message=f"Updated using post {post_id} with view_time {view_time}s"
        )

schema = strawberry.Schema(query=Query, mutation=Mutation)

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")