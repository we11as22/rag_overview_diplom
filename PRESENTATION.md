# Презентация — копипаст в слайды

---

## 1. Тема

**Сравнение стратегий retrieval в RAG и прототип агента поддержки**

Датасет: **Wix/WixQA** — 6 221 → **4 172** в индексе (без `feature_request`), **148** single-article QA  
Метрики: **MRR@10** (чанки), **Hit@10** (заголовки) · 3 embed-модели, sweep α и k · **без chain-of-RAG**

---

## 2. Эксперименты retrieval

```
corpus (4172) + QA (148)  →  BM25 | vector | hybrid (linear, RRF)
                                    ↓
                         3 embed-модели (Ollama)
                                    ↓
                         sweep α, k ∈ {1,3,5,6,8,10,15,20}
                                    ↓
                         лучший конфиг → rag_service + ADK tools
```

| Задача | Метод | Модель | α | MRR@10 | Recall@10 / Hit@10 |
|--------|--------------|--------|---|--------|---------------------|
| **Чанки** | hybrid linear | embeddinggemma | **0.7** | **0.577** | R@10 **0.865** |
| чанки | vector | embeddinggemma | — | 0.543 | R@10 0.872 |
| чанки | BM25 | — | — | 0.341 | R@10 0.547 |
| **Заголовки** | vector | embeddinggemma | **1.0** | 0.463* | Hit@10 **0.743** |
| заголовки | BM25 title | — | — | 0.180* | Hit@10 0.291 |

\* MRR@20 / Hit@20 = 0.818 для vector title

**В агенте:** chunks α=**0.7**, k=**8** (k=15 — 95% recall) · titles α=**1.0**, k=**10** · **embeddinggemma**  
Полные таблицы: `01_retrieval_eval/results/` · обоснование: `RETRIEVAL_SELECTION.md`

---

## 3. RAG в проде (кратко)

```
документ → параграфный чанк (~600 сим) → PG (без feature_request)
запрос   → Ollama embed → ts_rank_cd + pgvector → linear fusion
title hit → best_chunk (DISTINCT ON по distance)
```

---

## 4. Архитектура агента

```
Browser :8000
    ▼
┌─ agent_service (ADK + LiteLLM) ─────────────────┐
│ search_by_titles (α=1.0, k=10)                  │
│ search_by_chunks (α=0.7, k=8)                   │
│ open_article │ workspace_* │ memory              │
│ after_tool: spill >8K → agent_workspace         │
│ before_model: prune + LLM summary               │
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
