# linux/amd64 required on Apple Silicon: google-genai arm64 wheels can SIGILL (exit 132).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN adduser --disabled-password --gecos "" myuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R myuser:myuser /app

USER myuser
ENV PATH="/home/myuser/.local/bin:$PATH"

EXPOSE 8000
ENV PORT=8000
ENV ADK_DISABLE_LOCAL_STORAGE=1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
