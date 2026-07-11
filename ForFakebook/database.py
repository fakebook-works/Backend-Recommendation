import os
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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
