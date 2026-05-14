# RAG Retrieval Evaluation

Сравнение стратегий retrieval для дипломной работы. Датасет — [Wix/WixQA](https://huggingface.co/datasets/Wix/WixQA).

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

# 4. Запустить оценку
python run_eval.py

# Результаты: results/report.md, results/results.json

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
