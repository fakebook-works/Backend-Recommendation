import sys
import os
import numpy as np

# Add the current directory to sys.path so we can import EmbeddingModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EmbeddingModel import (
    generate_multimodal_embedding,
    SessionLocal,
    save_embedding,
    save_user_embedding,
    parse_vector
)
from sqlalchemy import text

def main():
    print("=== EMBEDDING TEST SCRIPT ===")
    
    title = input("Enter post title/content: ").strip()
    if not title:
        print("Error: Title/content cannot be empty.")
        return
        
    media_input = input("Enter image/video URLs (comma-separated, leave blank if none): ").strip()
    image_urls = [url.strip() for url in media_input.split(",") if url.strip()] if media_input else []
    
    print("\nGenerating post embedding (this might take a few seconds)...")
    post_vector = generate_multimodal_embedding(title, image_urls)
    
    print(f"\nSuccessfully generated post vector (Length: {len(post_vector)}).")
    print(f"Preview (first 10 elements): {post_vector[:10]}...")
    
    # Try to save to DB
    post_id = 999999  # Temporary test post ID
    if SessionLocal is not None:
        db = SessionLocal()
        try:
            print(f"Saving post embedding to database for post_id: {post_id}...")
            save_embedding(db, post_id, post_vector)
            print("Post embedding saved successfully.")
        except Exception as e:
            print(f"Error saving to DB: {e}")
            db.close()
            return
    else:
        print("Warning: Database not connected. Skipping DB save.")
        db = None

    print("\n=== USER EMBEDDING UPDATE ===")
    try:
        user_id = int(input("Enter user_id (integer): ").strip())
        view_time = float(input("Enter view_time in seconds: ").strip())
    except ValueError:
        print("Error: Invalid user_id or view_time.")
        if db:
            db.close()
        return

    if db is not None:
        try:
            print(f"\nUpdating user embedding for user_id: {user_id} based on post_id: {post_id}...")
            
            # Fetch existing user embedding
            user_row = db.execute(
                text("SELECT embedding FROM user_embeddings WHERE user_id = :user_id"),
                {"user_id": user_id}
            ).fetchone()
            
            post_emb = np.array(post_vector)
            if user_row:
                user_emb = parse_vector(user_row[0])
                w = min(view_time / 10.0, 2.0)
                new_emb = user_emb + w * post_emb
                new_emb = new_emb / np.linalg.norm(new_emb)
                print("Existing user embedding found. Performed weighted update.")
            else:
                new_emb = post_emb
                print("No existing user embedding. Initialized with post embedding.")
                
            save_user_embedding(db, user_id, new_emb.tolist())
            print(f"User embedding updated in database for user_id: {user_id}.")
            print(f"Preview of user vector (first 10 elements): {new_emb[:10].tolist()}...")
        except Exception as e:
            print(f"Error updating user embedding: {e}")
        finally:
            db.close()
    else:
        # Local simulation without database
        print("\n[Simulating without Database]")
        post_emb = np.array(post_vector)
        # Mock old user embedding as random
        old_user_emb = np.random.randn(512)
        old_user_emb = old_user_emb / np.linalg.norm(old_user_emb)
        w = min(view_time / 10.0, 2.0)
        new_emb = old_user_emb + w * post_emb
        new_emb = new_emb / np.linalg.norm(new_emb)
        print("Simulated weighted update on a random user embedding.")
        print(f"Preview of simulated user vector (first 10 elements): {new_emb[:10].tolist()}...")

if __name__ == "__main__":
    main()
