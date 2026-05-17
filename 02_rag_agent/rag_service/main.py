"""RAG Service — FastAPI app.

Endpoints:
  POST /ingest          — load corpus from WixQA jsonl files
  POST /search/hybrid   — hybrid search over documents (title-level)
  POST /search/chunks   — hybrid search over chunks (content-level)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from db import engine, Base
from routers import ingest, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # Tables already created by init.sql, but ensure vector dim comment exists
        pass
    yield
    await engine.dispose()


app = FastAPI(title="RAG Service", lifespan=lifespan)
app.include_router(ingest.router, prefix="/ingest")
app.include_router(search.router, prefix="/search")


@app.get("/health")
async def health():
    return {"status": "ok"}
