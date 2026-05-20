"""Фильтрация корпуса WixQA перед индексацией и бенчмарком."""
from __future__ import annotations

from typing import List

# Не how-to: «функция недоступна»; в QA почти не встречается как эталон
EXCLUDED_ARTICLE_TYPES = frozenset({"feature_request"})


def filter_corpus(docs: List[dict]) -> List[dict]:
    return [d for d in docs if d.get("article_type") not in EXCLUDED_ARTICLE_TYPES]
