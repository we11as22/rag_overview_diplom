#!/usr/bin/env bash
# Обёртка: полный запуск (rag + agent) — скрипт в корне проекта.
exec "$(cd "$(dirname "$0")/.." && pwd)/start.sh" "$@"
