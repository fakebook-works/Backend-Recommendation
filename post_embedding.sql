CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE post_embeddings (
post_id BIGINT PRIMARY KEY,

```
embedding VECTOR(768) NOT NULL,

model_name VARCHAR(100) NOT NULL,

created_at TIMESTAMPTZ DEFAULT NOW(),

updated_at TIMESTAMPTZ DEFAULT NOW()
```

);

COMMENT ON TABLE post_embeddings IS
'Stores semantic embeddings generated from post title and content';

COMMENT ON COLUMN post_embeddings.post_id IS
'Post identifier';

COMMENT ON COLUMN post_embeddings.embedding IS
'Vector representation of post content';

COMMENT ON COLUMN post_embeddings.model_name IS
'Embedding model used to generate vector';
