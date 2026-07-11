import numpy as np
from sqlalchemy import text
from database import parse_vector

def recommend_feed_logic(db, user_id: int, skip: int = 0, take: int = 20):
    # Step 1: Candidate Generation
    # 1a. Following candidates (author_id is followed by user_id)
    following_rows = db.execute(
        text("""
            SELECT content_id FROM fb.content_post 
            WHERE author_id IN (
                SELECT id2 FROM fb.assoc 
                WHERE id1 = :user_id 
                  AND assoc_type = 2 
                  AND is_deleted = false
            ) AND is_deleted = false
            ORDER BY created_at DESC LIMIT 100
        """),
        {"user_id": user_id}
    ).fetchall()
    following_ids = [row[0] for row in following_rows]
    
    # 1b. Popular candidates (by likes and comments interaction count)
    popular_rows = db.execute(
        text("""
            SELECT id2 AS content_id, COUNT(*) AS interaction_count
            FROM fb.assoc_inverse
            WHERE id2_type = 3
              AND assoc_type IN (3, 5)
              AND is_deleted = false
            GROUP BY id2
            ORDER BY interaction_count DESC LIMIT 100
        """)
    ).fetchall()
    popular_ids = [row[0] for row in popular_rows]
    popular_counts = {row[0]: row[1] for row in popular_rows}
    
    # 1c. Recent candidates
    recent_rows = db.execute(
        text("""
            SELECT content_id FROM fb.content_post 
            WHERE is_deleted = false 
            ORDER BY created_at DESC LIMIT 100
        """)
    ).fetchall()
    recent_ids = [row[0] for row in recent_rows]
    
    # Merge candidates (remove duplicates)
    candidate_ids = list(set(following_ids + popular_ids + recent_ids))
    if not candidate_ids:
        return []
        
    candidate_ids = candidate_ids[:300]
    
    # Step 2: Load User Embedding
    user_row = db.execute(
        text("SELECT embedding FROM user_embeddings WHERE user_id = :user_id"),
        {"user_id": user_id}
    ).fetchone()
    
    if user_row:
        user_emb = parse_vector(user_row[0])
    else:
        # Default fallback to random vector
        user_emb = np.random.randn(512)
        user_emb = user_emb / np.linalg.norm(user_emb)
        
    # Step 3: Load Post Embeddings
    post_rows = db.execute(
        text("SELECT post_id, embedding FROM post_embeddings WHERE post_id = ANY(:candidate_ids)"),
        {"candidate_ids": candidate_ids}
    ).fetchall()
    
    post_embs = {}
    for row in post_rows:
        post_embs[row[0]] = parse_vector(row[1])
        
    # Step 4 & 5: Calculate Semantic Similarity & Social / Hybrid Ranking
    ranked_posts = []
    for pid in candidate_ids:
        p_emb = post_embs.get(pid)
        if p_emb is not None:
            # Vectors are L2 normalized, so Cosine Similarity is simple dot product
            semantic_score = float(np.dot(user_emb, p_emb))
        else:
            semantic_score = 0.0
            
        # Social Score
        # Boost if author is followed (+1.0)
        is_following = 1.0 if pid in following_ids else 0.0
        # Add interaction popularity normalized (capped at 50 likes/comments = 1.0)
        interaction_cnt = popular_counts.get(pid, 0)
        popularity_score = min(interaction_cnt / 50.0, 1.0)
        
        social_score = 0.6 * is_following + 0.4 * popularity_score
        
        # Hybrid score: 0.6 * semantic + 0.4 * social
        final_score = 0.6 * semantic_score + 0.4 * social_score
        
        ranked_posts.append({
            "postId": pid,
            "score": final_score,
            "semanticScore": semantic_score,
            "socialScore": social_score
        })
        
    # Step 6: Sort and Return Top Posts
    ranked_posts.sort(key=lambda x: x["score"], reverse=True)
    paginated_posts = ranked_posts[skip:skip+take]
    
    # Save log outputs to rec_ranked_list and rec_ranked_item
    try:
        list_id = int(np.random.randint(1, 1000000000))
        db.execute(
            text("""
                INSERT INTO fb.rec_ranked_list (list_id, user_id, context, created_at)
                VALUES (:list_id, :user_id, 1, NOW())
            """),
            {"list_id": list_id, "user_id": user_id}
        )
        for idx, post in enumerate(paginated_posts):
            db.execute(
                text("""
                    INSERT INTO fb.rec_ranked_item (list_id, item_id, item_type, final_score, rank_pos)
                    VALUES (:list_id, :item_id, 3, :final_score, :rank_pos)
                """),
                {
                    "list_id": list_id,
                    "item_id": post["postId"],
                    "final_score": post["score"],
                    "rank_pos": skip + idx + 1
                }
            )
        db.commit()
    except Exception as e:
        print(f"Warning: Failed to save ranked list logs: {e}")
        db.rollback()
        
    return paginated_posts
