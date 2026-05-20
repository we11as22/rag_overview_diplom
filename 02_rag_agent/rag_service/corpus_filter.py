"""Фильтрация корпуса при ingest и search."""
from __future__ import annotations

EXCLUDED_ARTICLE_TYPES = frozenset({"feature_request"})

# SQL-фрагмент для documents d
DOC_TYPE_SQL = "COALESCE(d.article_type, 'article') NOT IN ('feature_request')"
