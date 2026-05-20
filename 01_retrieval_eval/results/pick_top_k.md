# Подбор top_k для агента

**α чанков:** 0.7 (лучший MRR@10 в `report.md`)  
**Применено в агенте:** `search_by_chunks` top_k=**8** (операционный), `search_by_titles` top_k=**10** — см. `RETRIEVAL_SELECTION.md`

EVAL_KS в .env: `[1, 3, 5, 6, 8, 10, 15, 20]`



### Чанки (search_by_chunks)
config: `hybrid_linear` embed=`embeddinggemma` α=0.7

| k | recall | recall | recall | recall | recall | recall | recall | recall | mrr | mrr | mrr | mrr | mrr | mrr | mrr | mrr |
|---|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| | 0.432 | 0.703 | 0.790 | 0.804 | 0.831 | 0.865 | 0.926 | 0.953 | 0.432 | 0.548 | 0.567 | 0.569 | 0.573 | 0.577 | 0.581 | 0.583 |

### Заголовки (search_by_titles)
config: `vector_title` embed=`embeddinggemma` α=None

| k | hit | hit | hit | hit | hit | hit | hit | hit | mrr | mrr | mrr | mrr | mrr | mrr | mrr | mrr |
|---|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| | 0.345 | 0.513 | 0.622 | 0.655 | 0.696 | 0.743 | 0.804 | 0.818 | 0.345 | 0.422 | 0.447 | 0.453 | 0.459 | 0.463 | 0.468 | 0.469 |

## Рекомендация

| Инструмент | top_k (recall/hit ≥ 95% @max) | top_k (max MRR в лимите) |
|------------|----------------------------------|---------------------------|
| `search_by_chunks` | **15** | 15 |
| `search_by_titles` | **10** | 10 |

- chunks: recall@15=0.926 >= 95%×0.926@15
- titles: hit@10=0.743 >= 95%×0.743@10

## В агенте (`agent.py`)

| Tool | α | top_k (в коде) | top_k (95% критерий) |
|------|---|----------------|----------------------|
| `search_by_chunks` | 0.7 | **8** | 15 |
| `search_by_titles` | 1.0 | **10** | 10 |
