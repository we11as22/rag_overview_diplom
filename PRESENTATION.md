# Презентация — копипаст в слайды

---

## 1. Тема

**Сравнение стратегий retrieval в RAG и прототип агента поддержки**

Датасет: **Wix/WixQA** — 6 221 статья, **148** QA (single-article)  
Метрика: **MRR@10** (чанки), **Hit@10** (заголовки)

---

## 2. Подбор поиска

```
corpus + QA  →  BM25 | vector | hybrid (linear, RRF)
                      ↓
              3 embed-модели (Ollama)
                      ↓
              sweep α, MRR@10 / Hit@10
                      ↓
              лучший конфиг → rag_service + tools агента
```

| Задача | Лучший метод | Модель | α | MRR@10 | Recall@10 / Hit@10 |
|--------|--------------|--------|---|--------|---------------------|
| **Чанки** (содержание) | hybrid linear | embeddinggemma | **0.6** | **0.57** | R@10 **0.86** |
| vector | embeddinggemma | — | 0.50 | R@10 0.86 |
| BM25 | — | — | 0.36 | R@10 0.59 |
| **Заголовки** (статья) | vector | embeddinggemma | **1.0** | 0.45 | Hit@10 **0.73** |
| BM25 titles | — | — | 0.16 | Hit@10 0.28 |

**В агенте:** `search_by_chunks` α=0.6 · `search_by_titles` α=1.0 · embed **embeddinggemma** 768d

---

## 3. RAG в проде (кратко)

```
документ → параграфный чанк (~600 сим) → PG
запрос   → Ollama embed → ts_rank_cd + pgvector → linear fusion
title hit → best_chunk (DISTINCT ON по distance), не обрезок текста
```

---

## 4. Архитектура агента

```
Browser :8000
    ▼
┌─ agent_service (ADK + LiteLLM) ─────────────────┐
│ search_by_titles │ search_by_chunks │ open_article │
│ workspace_list │ workspace_read │ workspace_search │
│ preload_memory │ load_memory                        │
│ after_tool: spill >8K → agent_workspace           │
│ before_model: prune + LLM summary истории         │
└──────────────────┬────────────────────────────────┘
                   │ :8001
┌─ rag_service ────┴────────────────────────────────┐
│ /search/hybrid · /search/chunks · /search/article │
└──────────────────┬────────────────────────────────┘
                   ▼
┌─ PostgreSQL ──────────────────────────────────────┐
│ documents · chunks · agent_workspace · agent_memory│
└───────────────────────────────────────────────────┘
        Ollama :11434 (хост)
```

---

## 5. Контекст агента

```
большой tool output  →  Postgres (построчно)  →  stub в промпт
длинная история      →  head + summary + tail  (~60K порог)
delete сессии        →  agent_memory (FTS)
```

**Стек:** ADK · LiteLLM · FastAPI · pgvector · Docker (PG) · `./start.sh`
