# Recommendation & Ranking Domain

Tài liệu này giải thích các bảng trong `recommendation_ranking_schema.sql` và cách vận hành hệ gợi ý/xếp hạng.

## Mục tiêu
Domain Recommendation & Ranking lưu tập ứng viên (candidate set), điểm số đặc trưng và danh sách xếp hạng cuối cho từng ngữ cảnh (feed, people you may know...).

## Các bảng

### `fb.rec_candidate_set`
Một “batch” ứng viên cho 1 user và 1 ngữ cảnh.
- **set_id**: Khóa chính.
- **user_id**: Người nhận gợi ý.
- **context**: Ngữ cảnh (`1=feed`, `2=people_you_may_know`, `3=groups`, `4=pages`).
- **created_at**: Thời điểm tạo batch.
- **data**: Metadata (model version, experiment, source).

### `fb.rec_candidate`
Danh sách ứng viên trong batch.
- **set_id**: Tham chiếu `rec_candidate_set`.
- **item_id / item_type**: Đối tượng được đề xuất.
- **base_score**: Điểm sơ bộ.
- **features**: JSON chứa feature phục vụ ranking.

### `fb.rec_ranked_list`
Danh sách đã xếp hạng cuối cùng.
- **list_id**: Khóa chính.
- **user_id**: Người nhận.
- **context**: Ngữ cảnh.
- **created_at**: Thời điểm tạo danh sách.
- **data**: Metadata (model version, latency, debug info).

### `fb.rec_ranked_item`
Các item trong danh sách xếp hạng.
- **list_id**: Tham chiếu `rec_ranked_list`.
- **item_id / item_type**: Đối tượng.
- **final_score**: Điểm cuối.
- **rank_pos**: Vị trí xếp hạng.

## Indexes
- `rec_candidate_user_idx`: Lấy candidate set theo user và thời gian.
- `rec_ranked_user_idx`: Lấy ranked list theo user và thời gian.

## Luồng tiêu biểu

### Gợi ý feed
1. Tạo `rec_candidate_set` cho user.
2. Ghi các candidate vào `rec_candidate` kèm features.
3. Ranking tạo `rec_ranked_list` + `rec_ranked_item`.
4. Feed đọc danh sách đã xếp hạng.

## Ghi chú triển khai
- Feature và model version nên lưu trong `data` để truy vết.
- DB thường chỉ lưu kết quả; pipeline ML xử lý ngoài DB.

