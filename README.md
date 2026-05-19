# RAG Overview — Дипломный проект

Сравнение стратегий retrieval для RAG-систем на датасете Wix WixQA.  
Реализован полный цикл: бенчмарк методов поиска → прототип диалогового агента на найденном лучшем конфиге.

---

## Структура репозитория

```
rag_overview_diplom/
├── data/                          # датасет (не в git, скачивается скриптом)
│   ├── corpus.jsonl               # 6 221 статья Wix Help Center
│   └── qa_expertwritten.jsonl     # 200 QA-пар с эталонными ответами
│
├── 01_retrieval_eval/             # модуль 1: бенчмарк методов поиска
│   ├── .env.example
│   ├── requirements.txt
│   ├── download_data.py           # скачать датасет с HuggingFace
│   ├── config.py
│   ├── embeddings.py              # async Ollama embedder
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── vector_retriever.py    # ChromaDB + Ollama
│   │   ├── hybrid_retriever.py    # linear fusion + RRF
│   │   └── chain_rag_retriever.py # LLM chain-of-RAG
│   ├── evaluate.py                # MRR@K, Recall@K, Precision@K, Hit@K
│   ├── run_eval.py                # полный бенчмарк (чанковый поиск)
│   ├── eval_title_search.py       # бенчмарк поиска по заголовкам
│   └── results/
│       ├── report.md              # результаты чанкового поиска
│       └── title_search_report.md # результаты поиска по заголовкам
│
└── 02_rag_agent/                  # модуль 2: прототип агента
    ├── docker-compose.yml         # только PostgreSQL
    ├── .env.example
    ├── ingest.py                  # залить корпус в БД
    ├── start.sh                   # запустить rag_service + agent локально
    ├── rag_service/               # FastAPI: embedding + hybrid search
    │   ├── Dockerfile
    │   ├── main.py
    │   ├── db.py                  # SQLAlchemy + pgvector
    │   ├── embedder.py            # Ollama client
    │   └── routers/
    │       ├── ingest.py          # параграфный чанкинг + загрузка в БД
    │       └── search.py          # /search/hybrid, /search/chunks, /search/article
    └── agent_service/             # Google ADK агент
        ├── Dockerfile
        ├── main.py                # get_fast_api_app + uvicorn (официальный паттерн)
        ├── requirements.txt
        ├── memory_service.py      # PostgresMemoryService (персистентная память)
        ├── services.py            # регистрация pgmemory:// схемы в ADK
        └── rag_agent/
            ├── agent.py           # root_agent + 7 инструментов
            └── __init__.py
```

---

## Архитектура системы

```
Пользователь (браузер)
        │
        │ http://localhost:8000  (ADK Web UI — встроенный чат)
        ▼
┌─────────────────────────────────────────────────┐
│  agent_service  (Google ADK 1.18)               │
│                                                 │
│  root_agent (LiteLlm → OpenAI-compatible API)  │
│                                                 │
│  Инструменты поиска:                            │
│   • search_by_titles   — топ-10 статей          │
│   • search_by_chunks   — топ-6 чанков           │
│   • open_article       — полный текст → артефакт│
│                                                 │
│  Инструменты артефактов:                        │
│   • list_saved_articles                         │
│   • read_article_lines — чтение по строкам      │
│   • search_in_article  — regex по артефакту     │
│   • search_in_articles — regex по нескольким    │
│                                                 │
│  Память: preload_memory_tool + load_memory_tool │
└─────────────┬───────────────────────────────────┘
              │ HTTP  (localhost:8001)
              ▼
┌─────────────────────────────────────────────────┐
│  rag_service  (FastAPI + uvicorn)               │
│                                                 │
│  POST /search/hybrid   — поиск по заголовкам    │
│  POST /search/chunks   — поиск по чанкам        │
│  GET  /search/article  — полный текст           │
│  POST /ingest          — загрузка корпуса       │
│                                                 │
│  Embedder: Ollama (embeddinggemma, порт 11434)  │
│  Fusion: BM25 (tsvector) + vector (pgvector)    │
│  Alpha: 0.6 (лучший по MRR@10)                 │
└─────────────┬───────────────────────────────────┘
              │ asyncpg
              ▼
┌─────────────────────────────────────────────────┐
│  PostgreSQL 16 + pgvector  (Docker)             │
│                                                 │
│  documents: id, title, contents,                │
│             title_vec vector(768), fts tsvector │
│                                                 │
│  chunks:    doc_id, chunk_text,                 │
│             chunk_vec vector(768), fts tsvector │
│                                                 │
│  agent_memory: персистентная память агента      │
└─────────────────────────────────────────────────┘

Ollama (локально, порт 11434) — НЕ в Docker
  Models: embeddinggemma (768d)
```

---

## Быстрый старт (рекомендуется: Ollama на хосте)

### Требования

- [Ollama](https://ollama.ai) на хосте: `ollama pull embeddinggemma`
- Docker Desktop — только для PostgreSQL
- Python 3.12+

---

### Шаг 1. Клонировать и настроить `.env`

```bash
git clone <repo>
cd rag_overview_diplom/02_rag_agent

cp .env.example .env
# Заполнить: OPENAI_API_BASE, OPENAI_API_KEY, LLM_MODEL
```

---

### Шаг 2. Скачать датасет (один раз, ~60 МБ)

```bash
cd ../01_retrieval_eval
pip install datasets
python download_data.py
cd ../02_rag_agent
```

---

### Шаг 3. PostgreSQL в Docker, rag + agent локально

```bash
docker compose up -d postgres
./start.sh
```

Открыть: **http://localhost:8000** (ADK UI)

Ollama и эмбеддинги — как раньше, на хосте (`OLLAMA_BASE_URL=http://localhost:11434` в `.env`).

---

### Шаг 4. Загрузить корпус в PostgreSQL (один раз, ~30–60 мин)

```bash
pip install httpx
python ingest.py
# Прогресс в stdout ingest.py (rag_service на :8001)
```

---

### Опционально: rag + agent в Docker

```bash
docker compose up --build
python ingest.py --docker-rag          # корпус читается внутри контейнера rag
```

Ollama для `rag_service` в compose: `OLLAMA_BASE_URL_DOCKER` (по умолчанию `host.docker.internal`).  
В `.env` оставь `OLLAMA_BASE_URL=http://localhost:11434` — он для `./start.sh`, не подмешивается в rag-контейнер.  
На **Apple Silicon** `agent_service` — `linux/amd64` (exit **132** на arm64).  
Если поиск 500 (Ollama недоступен из Docker) — используй `./start.sh`.

### Паритет Docker / локально

| Функция | `./start.sh` + postgres | `docker compose` (всё в контейнерах) |
|--------|-------------------------|--------------------------------------|
| Поиск title/chunk, open_article | ✓ | ✓ если Ollama доступен с хоста |
| workspace_* (PostgreSQL) | ✓ | ✓ |
| preload/load memory | ✓ | ✓ |
| Удаление сессии → память + чистка workspace | ✓ | ✓ |
| ADK Web UI + chat SSE | ✓ | ✓ |
| Ingest корпуса | `python ingest.py` | `python ingest.py --docker-rag` |
| Граф агента в dev-ui (`/dev/build_graph`) | 500 (LiteLlm) | то же — не ломает чат |

Сессии ADK (`pgclean`) — **in-memory до перезапуска** агента; в Postgres остаются workspace, memory после delete сессии.

### Логи (режим full Docker)

```bash
docker compose logs -f                    # все сервисы
docker compose logs -f agent_service      # только агент
docker compose logs -f rag_service        # только RAG API
```

### Остановка

```bash
docker compose down        # остановить, данные сохранить
docker compose down -v     # остановить + удалить данные PostgreSQL
```

---

## Модуль 1: бенчмарк методов поиска

### Запуск полного бенчмарка

```bash
cd 01_retrieval_eval
cp .env.example .env
# Заполнить OLLAMA_EMBED_MODELS, OLLAMA_BASE_URL

python download_data.py          # скачать данные
python run_eval.py --skip-chain  # бенчмарк BM25/vector/hybrid
python eval_title_search.py      # бенчмарк поиска по заголовкам
```

Результаты: `results/report.md`, `results/title_search_report.md`

### Лучшие конфигурации

**Поиск по чанкам (содержанию):**

| Метод | Модель | Alpha | MRR@10 | Recall@10 |
|---|---|---|---|---|
| hybrid_linear | embeddinggemma | 0.6 | **0.5703** | 0.8581 |
| hybrid_rrf | embeddinggemma | w=0.7 | 0.5196 | 0.8649 |
| vector | embeddinggemma | — | 0.5007 | 0.8649 |
| BM25 | — | — | 0.3605 | 0.5878 |

**Поиск по заголовкам:**

| Метод | Модель | Alpha | Hit@10 | MRR@10 |
|---|---|---|---|---|
| vector_title | embeddinggemma | — | **0.7297** | 0.4531 |
| hybrid_title | embeddinggemma | 0.9 | 0.7162 | 0.4410 |
| hybrid_title | embeddinggemma | 0.6 | 0.7095 | 0.4008 |

*Оценка на 148 single-article QA парах из 200.*

---

## Переменные окружения

### `02_rag_agent/.env`

| Переменная | Описание | Пример |
|---|---|---|
| `LLM_API_BASE` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API ключ | `sk-...` |
| `LLM_MODEL` | Имя модели | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | URL Ollama | `http://localhost:11434` |
| `EMBED_MODEL` | Модель эмбеддингов | `embeddinggemma` |
| `EMBED_DIM` | Размерность эмбеддингов | `768` |

### `01_retrieval_eval/.env`

| Переменная | Описание |
|---|---|
| `OLLAMA_BASE_URL` | URL Ollama |
| `OLLAMA_EMBED_MODELS` | Модели через запятую |
| `HYBRID_ALPHAS` | Значения alpha для sweep |
| `EVAL_KS` | Значения K для метрик |

---

## Пересборка rag_service (после изменений кода)

```bash
cd 02_rag_agent
docker stop rag_service_new; docker rm rag_service_new
docker build -t rag_service_img ./rag_service
docker run -d --name rag_service_new --network 02_rag_agent_default -p 8001:8001 -v /path/to/data:/data:ro -e DATABASE_URL="postgresql+asyncpg://rag:rag@02_rag_agent-postgres-1:5432/rag" -e OLLAMA_BASE_URL="http://host.docker.internal:11434" -e EMBED_MODEL="embeddinggemma" -e EMBED_DIM="768" rag_service_img
```

---

## Технологии

| Компонент | Технология |
|---|---|
| Векторная БД | PostgreSQL 16 + pgvector |
| Полнотекстовый поиск | PostgreSQL `tsvector` / `ts_rank_cd` |
| Эмбеддинги | Ollama (`embeddinggemma`, 768d) |
| RAG API | FastAPI + SQLAlchemy async |
| Агент | Google ADK 1.18 + LiteLLM |
| Веб-UI | ADK встроенный чат (SSE streaming) |
| Память агента | PostgreSQL (кастомный `BaseMemoryService`) |
| Артефакты | ADK `file://` artifact service |
| Оркестрация | Docker Compose (только PostgreSQL) |
