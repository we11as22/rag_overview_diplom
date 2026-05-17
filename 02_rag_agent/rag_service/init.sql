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

-- Agent workspace: построчное хранилище выводов инструментов
-- Область видимости: session_id, TTL 7 дней
CREATE TABLE IF NOT EXISTS agent_workspace (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    key         TEXT        NOT NULL,
    tool_name   TEXT,
    line_number INT         NOT NULL,
    line_text   TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (session_id, key, line_number)
);

CREATE INDEX IF NOT EXISTS ws_session_key_idx ON agent_workspace(session_id, key);
CREATE INDEX IF NOT EXISTS ws_key_line_idx    ON agent_workspace(session_id, key, line_number);
CREATE INDEX IF NOT EXISTS ws_fts_idx         ON agent_workspace USING GIN(to_tsvector('english', line_text));
CREATE INDEX IF NOT EXISTS ws_created_at_idx  ON agent_workspace(created_at);

-- Agent memory: персистентная память между сессиями
CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGSERIAL PRIMARY KEY,
    app_name    TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT,
    author      TEXT,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_json TEXT NOT NULL,
    fts         TSVECTOR GENERATED ALWAYS AS (
                    to_tsvector('english', content_json)
                ) STORED
);

CREATE INDEX IF NOT EXISTS agent_memory_user_idx ON agent_memory(app_name, user_id);
CREATE INDEX IF NOT EXISTS agent_memory_fts_idx  ON agent_memory USING GIN(fts);
