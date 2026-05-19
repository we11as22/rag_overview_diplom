#!/usr/bin/env bash
# Запускает rag_service и agent_service локально (не в Docker).
# Postgres при этом должен быть поднят: docker compose up -d
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Нет .env — скопируй из .env.example и заполни"
  exit 1
fi

# Загрузить .env
set -a; source "$ENV_FILE"; set +a

export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://rag:rag@localhost:5432/rag}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
export EMBED_MODEL="${EMBED_MODEL:-embeddinggemma}"
export EMBED_DIM="${EMBED_DIM:-768}"
export RAG_SERVICE_URL="${RAG_SERVICE_URL:-http://localhost:8001}"
export LLM_MODEL="${LLM_MODEL:?Задай LLM_MODEL в .env}"
export OPENAI_API_BASE="${LLM_API_BASE:?Задай LLM_API_BASE в .env}"
export OPENAI_API_KEY="${LLM_API_KEY:?Задай LLM_API_KEY в .env}"
export PORT="${PORT:-8000}"
export SESSION_SERVICE_URI="${SESSION_SERVICE_URI:-pgclean://localhost}"
export MEMORY_SERVICE_URI="${MEMORY_SERVICE_URI:-pgmemory://localhost}"
export ADK_DISABLE_LOCAL_STORAGE="${ADK_DISABLE_LOCAL_STORAGE:-1}"

echo "==> Запускаем rag_service на :8001 ..."
cd "$SCRIPT_DIR/rag_service"
pip install -q -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 &
RAG_PID=$!

# Ждём пока rag_service поднимется
echo -n "Ждём rag_service"
for i in $(seq 1 20); do
  sleep 1
  if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo " OK"
    break
  fi
  echo -n "."
done

echo "==> Запускаем agent_service (uvicorn) на :8000 ..."
cd "$SCRIPT_DIR/agent_service"
pip install -q -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port "$PORT" &
AGENT_PID=$!

echo ""
echo "✓ rag_service  → http://localhost:8001"
echo "✓ agent web UI → http://localhost:8000"
echo ""
echo "Ctrl+C чтобы остановить всё"

trap "kill $RAG_PID $AGENT_PID 2>/dev/null; exit" INT TERM
wait
