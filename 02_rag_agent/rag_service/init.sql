CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Full documents with title embedding (for hybrid title search)
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    url         TEXT,
    title       TEXT NOT NULL DEFAULT '',
    contents    TEXT NOT NULL DEFAULT '',
    article_type TEXT,
    title_vec   vector(768),
    fts         tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(contents, ''))
                ) STORED
);

CREATE INDEX IF NOT EXISTS documents_fts_idx ON documents USING GIN(fts);
CREATE INDEX IF NOT EXISTS documents_title_vec_idx ON documents USING hnsw(title_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Chunks of document contents (for deep semantic search)
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT  NOT NULL,
    chunk_text  TEXT NOT NULL,
    chunk_vec   vector(768),
    fts         tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', chunk_text)
                ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_fts_idx    ON chunks USING GIN(fts);
CREATE INDEX IF NOT EXISTS chunks_vec_idx    ON chunks USING hnsw(chunk_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
