"""ADK services.py — регистрирует PostgresMemoryService.

ADK автоматически загружает этот файл при старте adk web.
Регистрируем схему 'pgmemory://' чтобы передать через --memory_service_uri.
"""
from google.adk.cli.service_registry import get_service_registry

from memory_service import PostgresMemoryService


def _postgres_memory_factory(uri: str, **kwargs):
    """Фабрика для PostgresMemoryService. URI игнорируется — DSN берётся из DATABASE_URL."""
    return PostgresMemoryService()


get_service_registry().register_memory_service("pgmemory", _postgres_memory_factory)
