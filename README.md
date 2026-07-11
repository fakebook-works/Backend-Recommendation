# AI Recommendation & Embedding Service

Dịch vụ vi mô (microservice) AI sinh vector embedding đa phương thức (CLIP) và xếp hạng cá nhân hóa bảng tin (Hybrid Recommendation) cho mạng xã hội Fakebook.

---

## 📂 Cấu trúc dự án (`ForFakebook/`)

Dự án được tái cấu trúc theo dạng modular:

```text
ForFakebook/
├── EmbeddingModel.py           # Entrypoint chính (FastAPI & GraphQL Router)
├── database.py                 # Cấu hình PostgreSQL & SQLAlchemy persistence helpers
├── embedding_service.py        # Logic tải CLIP models & trích xuất vector đa phương thức
├── recommendation_service.py   # Triển khai thuật toán gợi ý 6 bước & Hybrid Ranking
├── user_embedding.sql          # Khởi tạo bảng lưu vector sở thích người dùng (Kiểu vector 512 chiều)
├── post_embedding.sql          # Khởi tạo bảng lưu vector nội dung bài viết
├── test_embedding.py           # Script test tương tác embedding thủ công
├── run_test_non_interactive.py # Script test tự động (Non-interactive)
└── README.md                   # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠 Yêu cầu hệ thống & Cài đặt

### 1. Yêu cầu tiên quyết
* **Python 3.10+**
* **PostgreSQL** có cài đặt extension [pgvector](https://github.com/pgvector/pgvector) (hỗ trợ lưu trữ và tìm kiếm vector).

### 2. Cài đặt thư viện Python
Cài đặt các gói phụ thuộc cần thiết:
```bash
pip install fastapi uvicorn sentence-transformers sqlalchemy psycopg2-binary pillow requests numpy strawberry-graphql opencv-python
```

### 3. Setup Database
Chạy 2 file SQL sau vào cơ sở dữ liệu PostgreSQL để kích hoạt extension vector và tạo các bảng lưu trữ:
```bash
psql -h localhost -U postgres -d fakebook -f user_embedding.sql
psql -h localhost -U postgres -d fakebook -f post_embedding.sql
```

---

## 🚀 Khởi chạy Service

Cấu hình biến môi trường kết nối Database (nếu khác mặc định):
* **Windows (PowerShell):**
  ```powershell
  $env:DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fakebook"
  ```
* **Linux/macOS:**
  ```bash
  export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/fakebook"
  ```

Chạy FastAPI server bằng `uvicorn`:
```bash
uvicorn EmbeddingModel:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📑 Danh sách API

Dịch vụ chạy song song cả **REST API** (dành cho Backend-to-Backend) và **GraphQL API** (dành cho Frontend).

### 1. REST API (Khởi tạo tài khoản mới)
Được gọi từ Main Backend khi có người dùng đăng ký mới để khởi tạo vector sở thích ngẫu nhiên:
* **Endpoint:** `POST http://localhost:8000/initialize-user/{user_id}`
* **Response (JSON):**
  ```json
  {
    "success": true,
    "user_id": 123,
    "message": "Random embedding initialized via REST API"
  }
  ```

### 2. GraphQL API (Gợi ý & Tương tác)
* **Endpoint:** `POST http://localhost:8000/graphql`

#### A. Query lấy Feed cá nhân hóa (`recommendFeed`)
Tìm kiếm 100-300 bài viết ứng viên (Following, Popular, Recent), so khớp cosine similarity với vector sở thích của user, tính Hybrid Score và trả về danh sách được xếp hạng:
```graphql
query GetRecommendedFeed($userId: Int!, $skip: Int!, $take: Int!) {
  recommendFeed(userId: $userId, skip: $skip, take: $take) {
    postId
    score
    semanticScore
    socialScore
  }
}
```

#### B. Mutation tạo vector bài viết mới (`createPostEmbedding`)
Được kích hoạt khi người dùng đăng bài viết mới để sinh vector văn bản + hình ảnh/video:
```graphql
mutation CreatePostEmbedding($postId: Int!, $title: String!, $imageUrls: [String!]!) {
  createPostEmbedding(postId: $postId, title: $title, imageUrls: $imageUrls) {
    success
    postId
  }
}
```

#### C. Mutation cập nhật sở thích khi xem bài viết (`updateUserEmbedding`)
Cập nhật lũy tiến vector sở thích của user dựa trên hành vi tương tác và thời gian xem bài đăng:
```graphql
mutation UpdateUser($userId: Int!, $postId: Int!, $viewTime: Float!) {
  updateUserEmbedding(userId: $userId, postId: $postId, viewTime: $viewTime) {
    success
    userId
    message
  }
}
```

---

## 💻 Tích hợp Frontend

Phía React Frontend tích hợp hai tính năng chính:
1. **Lấy Feed Gợi ý:** Tải Post IDs xếp hạng từ `recommendFeed` và thực hiện batch fetch chi tiết nội dung từ API Gateway. Có cơ chế tự động fallback về feed truyền thống nếu microservice AI gặp sự cố.
2. **Theo dõi thời gian xem (View tracking):** Sử dụng `IntersectionObserver` thông qua component `TrackedPostCard` để ghi nhận thời gian bài viết hiển thị trên màn hình của user. Nếu user dừng lại xem bài đăng trên 1 giây, khi cuộn qua sẽ tự động gọi mutation `updateUserEmbedding` để AI cập nhật lại hồ sơ sở thích của người dùng đó.
