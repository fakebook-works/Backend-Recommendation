# Recommendation System - Phase 2 (AI-Powered Semantic Recommendation)

## Overview

Phase 1 sử dụng Social Graph để tạo danh sách Candidate Posts.

Candidate được lấy từ:

* Following Posts
* Popular Posts
* Recent Posts

Sau khi có Candidate Posts, hệ thống sử dụng AI Embedding để thực hiện Semantic Re-ranking.

Mục tiêu:

Thay vì chỉ đề xuất dựa trên:

* Người dùng đang theo dõi ai
* Bài viết nào phổ biến

Hệ thống sẽ đề xuất dựa trên:

* Nội dung bài viết
* Sở thích của người dùng

---

# Database Schema

Phase 2 bổ sung 2 bảng:

## post_embeddings

Lưu vector biểu diễn nội dung bài viết.

Mỗi bài viết sẽ có một embedding duy nhất được sinh từ:

Title + Content

Ví dụ:

Post:
How to learn React effectively

↓

Embedding:

[0.12, -0.45, 0.81, ...]

---

## user_embeddings

Lưu vector biểu diễn sở thích của người dùng.

Embedding được tạo từ lịch sử tương tác:

* Follow
* Reaction
* Comment
* Save

được lưu trong bảng Associations.

Ví dụ:

User thích:

* AI
* Machine Learning
* Data Science

↓

Embedding:

[0.51, -0.14, 0.22, ...]

---

# Recommendation Flow

## Step 1

Candidate Generation

Thuật toán hiện tại thực hiện:

* Following Candidate
* Popular Candidate
* Recent Candidate

↓

Merge Candidate Posts

↓

Top 100-300 Candidates

---

## Step 2

Load User Embedding

Từ bảng:

user_embeddings

---

## Step 3

Load Post Embeddings

Từ bảng:

post_embeddings

---

## Step 4

Calculate Semantic Similarity

Semantic Score được tính bằng:

Cosine Similarity

giữa:

User Embedding

và

Post Embedding

---

## Step 5

Hybrid Ranking

Final Score được tính:

Final Score =
0.6 × Semantic Score
+
0.4 × Social Score

Trong đó:

Social Score:
được sinh từ thuật toán recommendation hiện tại.

Semantic Score:
được sinh từ cosine similarity.

---

## Step 6

Sort and Return Feed

Các bài viết được sắp xếp theo:

Final Score DESC

↓

Top 50 Posts

↓

For You Feed
