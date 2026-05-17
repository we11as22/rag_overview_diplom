FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# adk web <agents_dir> — запускаем из /app, агент лежит в /app/rag_agent/
CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8000", "/app"]
