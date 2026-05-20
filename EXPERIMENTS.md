# Отчёт по экспериментам retrieval

Датасет: [Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA).  
Дата прогона: май 2026. Полный цикл: `01_retrieval_eval/run_validation.py`.

**Почему выбраны метод, α и top_k:** [RETRIEVAL_SELECTION.md](RETRIEVAL_SELECTION.md)

## Данные и фильтры

| Параметр | Значение |
|----------|----------|
| Корпус в датасете | 6 221 документов |
| Корпус в индексе | **4 172** (`article` + `known_issue`, без `feature_request`) |
| QA всего | 200 |
| QA в оценке | **148** (single-article, один gold `doc_id`) |
| Чанкинг | параграфы ~600 символов |
| Embed (prod) | **embeddinggemma** 768d, Ollama |

**Ограничения прогона:** single-article QA (148/200); **chain-of-RAG не оценивался** (`--skip-chain`); сравнение 3 embed-моделей — в полных таблицах, в агенте только embeddinggemma.

## Эксперимент 1 — поиск по чанкам (содержание)

**Скрипт:** `run_eval.py` · **Индекс:** Chroma + BM25 · **Метрика:** MRR@K, Recall@K, P@K  
**Методы:** BM25, vector, hybrid linear (α ∈ [0, 0.1, …, 1.0]), hybrid RRF (w ∈ {0.3, 0.5, 0.7})  
**Модели:** embeddinggemma, qwen3-embedding:0.6b, bge-m3  
**K:** 1, 3, 5, 6, 8, 10, 15, 20

### Лучшие конфигурации (MRR@10)

| Rank | Метод | Модель | α | MRR@10 | Recall@10 |
|------|-------|--------|---|--------|-----------|
| 1 | hybrid linear | embeddinggemma | **0.7** | **0.577** | 0.865 |
| 2 | hybrid linear | embeddinggemma | 0.8 | 0.569 | 0.872 |
| 3 | hybrid linear | embeddinggemma | 0.6 | 0.562 | 0.865 |
| 6 | vector | embeddinggemma | — | 0.543 | 0.872 |
| 43 | BM25 | — | — | 0.341 | 0.547 |
| — | hybrid RRF | embeddinggemma | w=0.7 | 0.495 | 0.851 |

Полная таблица (46 конфигов × 8 k): `01_retrieval_eval/results/report.md`. RRF ниже linear hybrid при том же embed.

### Принятая конфигурация для агента

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Метод | hybrid linear | лучший MRR@10 среди всех вариантов |
| Модель | embeddinggemma | лидер на всех α |
| **α** | **0.7** | max MRR@10 = 0.577 |
| **top_k** | **8** | баланс качество/контекст: Recall@8 = 0.831, MRR@8 = 0.573 |
| top_k (альт.) | 15 | 95% от max Recall@15 при α=0.7 (`pick_top_k.py`) |

## Эксперимент 2 — поиск по заголовкам (статья целиком)

**Скрипт:** `eval_title_search.py` · **Поле:** title · **Метрика:** Hit@K, MRR@K  
**Методы:** BM25 title, vector title, hybrid title (α sweep)

### Лучшие конфигурации (MRR@20 / Hit@10)

| Метод | Модель | α | MRR@20 | Hit@10 | Hit@20 |
|-------|--------|---|--------|--------|--------|
| vector / hybrid | embeddinggemma | **1.0** | 0.469 | **0.743** | 0.818 |
| BM25 title | — | — | 0.187 | 0.290 | 0.385 |

Полная таблица: `01_retrieval_eval/results/title_search_report.md`.

### Принятая конфигурация для агента

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Метод | vector (α=1.0) | ≈ hybrid α=1.0, проще |
| Модель | embeddinggemma | Hit@10 = 0.743 |
| **α** | **1.0** | только dense по заголовкам |
| **top_k** | **10** | Hit@10 = 0.743 = 95% от max Hit@10 |

## Эксперимент 3 — подбор top_k

**Скрипт:** `pick_top_k.py --chunks-alpha 0.7`  
**Критерий:** ≥ 95% от метрики на максимальном k в `EVAL_KS`

| Инструмент | top_k (95%) | top_k (max MRR в лимите) | В агенте |
|------------|-------------|---------------------------|----------|
| `search_by_chunks` | 15 | 15 | **8** (операционный) |
| `search_by_titles` | 10 | 10 | **10** |

Детали: `01_retrieval_eval/results/pick_top_k.md`.

## Сводка: конфиг в production (`02_rag_agent`)

```
embeddinggemma (768d)
search_by_titles  →  α=1.0,  top_k=10
search_by_chunks  →  α=0.7,  top_k=8
corpus filter     →  без feature_request
fusion            →  min-max + linear (ts_rank_cd + pgvector <=>)
```

Файлы с константами:

- `02_rag_agent/agent_service/rag_agent/agent.py` — `_ALPHA_*`, defaults `top_k`
- `02_rag_agent/rag_service/embedder.py` — `EMBED_MODEL`
- `01_retrieval_eval/pick_top_k.py` — default `--chunks-alpha 0.7`

## Воспроизведение

```bash
cd 01_retrieval_eval
python download_data.py
python run_validation.py
# или: run_eval.py --skip-chain --force-rebuild && eval_title_search.py && pick_top_k.py --chunks-alpha 0.7
```
