# RAG Retrieval Evaluation

Сравнение стратегий retrieval для дипломной работы. Датасет — [Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA).

## Принятые конфиги (→ `02_rag_agent`)

| Инструмент | Метод | embed | α | top_k | MRR@10 / Hit@10 |
|------------|-------|-------|---|-------|-----------------|
| `search_by_chunks` | hybrid linear | embeddinggemma | **0.7** | **8** | MRR@10 **0.577** |
| `search_by_titles` | vector | embeddinggemma | **1.0** | **10** | Hit@10 **0.743** |

Корпус: **4172** док. (без `feature_request`), QA: **148**.  
Обоснование подбора: [../RETRIEVAL_SELECTION.md](../RETRIEVAL_SELECTION.md) · сводка: [../EXPERIMENTS.md](../EXPERIMENTS.md).

## Методы

| Метод | Описание |
|-------|----------|
| BM25 | Полнотекстовый поиск (rank-bm25) |
| Vector | Плотный поиск через эмбеддинги Ollama + ChromaDB |
| Hybrid | Линейная комбинация BM25 + Vector с перебором весов alpha |

## Метрики

- **MRR@K** (Mean Reciprocal Rank) — основная
- Recall@K, Precision@K

## Быстрый старт

```bash
cd 01_retrieval_eval

# 1. Настроить окружение
cp .env.example .env
# Заполнить .env (Ollama URL, модели)

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Скачать датасет (~60 MB)
python download_data.py

# 4. Полный цикл: валидация + top_k (чанки + заголовки)
python run_validation.py
# или по шагам:
# python run_eval.py --skip-chain --force-rebuild
# python eval_title_search.py
# python pick_top_k.py

# Результаты:
#   results/report.md, results.json          — чанки, все методы
#   results/title_search_report.md           — заголовки
#   results/pick_top_k.md                    — рекомендуемый top_k для агента

# Быстрее (одна модель):
# python run_validation.py --models embeddinggemma

# Для быстрой проверки (5 запросов):
python run_eval.py --dry-run

# Принудительно пересобрать индексы:
python run_eval.py --force-rebuild
```

## Требования

- Python 3.10+
- [Ollama](https://ollama.ai) запущен локально
- Нужные модели скачаны: `ollama pull gemma3-embedding`, `ollama pull qwen3-embedding:0.6b`, `ollama pull bge-m3`

## Структура

```
01_retrieval_eval/
├── .env.example          # Шаблон переменных окружения
├── requirements.txt
├── download_data.py      # Скачивает датасет в ../data/
├── config.py             # Конфиг из .env
├── embeddings.py         # OllamaEmbedder
├── retrieval/
│   ├── bm25_retriever.py
│   ├── vector_retriever.py
│   └── hybrid_retriever.py
├── evaluate.py           # Функции метрик
├── run_eval.py           # Оркестратор экспериментов
└── results/              # Создаётся автоматически
    ├── results.json
    └── report.md
```
