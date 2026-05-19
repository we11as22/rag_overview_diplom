# RAG Overview — диплом

Бенчмарк retrieval на **Wix/WixQA** → прототип агента на лучших конфигах.

```
01_retrieval_eval/   бенчмарк (Chroma + Ollama)
02_rag_agent/         rag_service + agent_service (PG + ADK)
data/                 corpus + QA (download_data.py)
```

---

## Подбор retrieval

**Данные:** 6 221 статья, **148** single-article QA · метрика **MRR@10** / **Hit@10**

```
QA + corpus → BM25 | vector | hybrid (linear, RRF) × 3 embed-модели → sweep α
```

| Задача | Метод | embed | α | MRR@10 | Recall@10 / Hit@10 |
|--------|-------|-------|---|--------|---------------------|
| чанки | **hybrid linear** | embeddinggemma | **0.6** | **0.570** | R@10 0.858 |
| чанки | vector | embeddinggemma | — | 0.501 | 0.865 |
| чанки | BM25 | — | — | 0.361 | 0.588 |
| заголовки | **vector** | embeddinggemma | **1.0** | 0.453 | Hit@10 **0.730** |
| заголовки | BM25 | — | — | 0.164 | 0.277 |

Полные таблицы: `01_retrieval_eval/results/report.md`, `title_search_report.md`

**В rag_service:** гибрид `ts_rank_cd` + pgvector `<=>`, min-max, linear fusion; чанкинг по параграфам ~600 сим; в title-search — **best_chunk** по distance.

**Запуск бенчмарка:**
```bash
cd 01_retrieval_eval && python download_data.py
python run_eval.py --skip-chain && python eval_title_search.py
```

---

## Агент

```
:8000 ADK UI
    ▼
agent_service ──HTTP──► rag_service ──► PostgreSQL (+ pgvector)
    │                         ▲
    │                    Ollama embeddinggemma
    ├─ search_by_titles (α=1.0)
    ├─ search_by_chunks (α=0.6)
    ├─ open_article → workspace
    ├─ workspace_list / read / search
    ├─ spill tool results >8K → agent_workspace
    ├─ compress history (~60K) → LLM summary
    └─ memory: delete session → agent_memory
```

| Слой | Что |
|------|-----|
| Поиск | конфиги из бенчмарка, два α |
| Workspace | Postgres, построчно, FTS по строкам |
| Контекст | `after_tool` spill, `before_model` сжатие |
| Память | `PostgresMemoryService`, preload/load |
| Сессии | `pgclean` in-memory до рестарта агента |

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
pip install httpx && python ingest.py
```

Ollama на хосте: `ollama pull embeddinggemma`.  
Full Docker / Apple Silicon / Ollama из контейнера — см. комментарии в `docker-compose.yml`.

**Стек:** PostgreSQL 16 + pgvector · FastAPI · Google ADK · LiteLLM · Ollama 768d
