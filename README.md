# RAG Overview — диплом

Бенчмарк retrieval на **Wix/WixQA** → прототип агента на лучших конфигах.

```
01_retrieval_eval/   бенчмарк (Chroma + Ollama)
02_rag_agent/         rag_service + agent_service (PG + ADK)
data/                 corpus + QA (download_data.py)
EXPERIMENTS.md        сводка экспериментов и цифры
RETRIEVAL_SELECTION.md  обоснование подбора метода, α и top_k
PRESENTATION.md       слайды (копипаст)
```

---

## Принятые конфигурации (после `run_validation.py`)

| Компонент | Значение |
|-----------|----------|
| Корпус | 4 172 док. (`article` + `known_issue`, без `feature_request`) |
| QA | 148 single-article |
| Embed | **embeddinggemma** 768d |
| `search_by_chunks` | hybrid linear, **α=0.7**, **top_k=8** |
| `search_by_titles` | vector, **α=1.0**, **top_k=10** |

Обоснование подбора: **[RETRIEVAL_SELECTION.md](RETRIEVAL_SELECTION.md)** · таблицы: **[EXPERIMENTS.md](EXPERIMENTS.md)**

---

## Метрики retrieval

| Задача | Метод | α | MRR@10 | Recall@10 / Hit@10 |
|--------|-------|---|--------|---------------------|
| чанки | **hybrid linear** | **0.7** | **0.577** | R@10 **0.865** |
| чанки | vector | — | 0.543 | 0.872 |
| чанки | BM25 | — | 0.341 | 0.547 |
| заголовки | **vector** | **1.0** | 0.463 (MRR@10) | Hit@10 **0.743** |
| заголовки | BM25 | — | 0.180 (@20) | 0.291 |

Полные таблицы: `01_retrieval_eval/results/report.md`, `title_search_report.md`, `pick_top_k.md`

**В rag_service:** гибрид `ts_rank_cd` + pgvector `<=>`, min-max, linear fusion; чанкинг ~600 сим; title-search — **best_chunk** по distance.

**Запуск бенчмарка:**
```bash
cd 01_retrieval_eval && python download_data.py
python run_validation.py
```

---

## Агент

```
:8000 ADK UI
    ▼
agent_service ──HTTP──► rag_service ──► PostgreSQL (+ pgvector)
    │                         ▲
    │                    Ollama embeddinggemma
    ├─ search_by_titles (α=1.0, top_k=10)
    ├─ search_by_chunks (α=0.7, top_k=8)
    ├─ open_article → workspace
    ├─ workspace_list / read / search
    ├─ spill tool results >8K → agent_workspace
    ├─ compress history (~60K) → LLM summary
    └─ memory: delete session → agent_memory
```

| Слой | Что |
|------|-----|
| Поиск | конфиги из `EXPERIMENTS.md` |
| Workspace | Postgres, построчно, FTS |
| Контекст | `after_tool` spill, `before_model` сжатие |
| Память | `PostgresMemoryService`, preload/load |

---

## Быстрый старт

```bash
# 1. данные
cd 01_retrieval_eval && python download_data.py && cd ../02_rag_agent

# 2. .env из .env.example (LLM_*, OLLAMA_*)

# 3. PG + сервисы
docker compose up -d postgres
./start.sh                    # http://localhost:8000

# 4. корпус (разово)
pip install httpx && python ingest.py --clear
```

Ollama на хосте: `ollama pull embeddinggemma`.  
Full Docker / Apple Silicon — см. `docker-compose.yml`.

**Стек:** PostgreSQL 16 + pgvector · FastAPI · Google ADK · LiteLLM · Ollama 768d
