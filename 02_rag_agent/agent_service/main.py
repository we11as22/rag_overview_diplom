"""FastAPI entry point for ADK agent (official Cloud Run / GKE pattern)."""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app

load_dotenv(override=False)

AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    session_service_uri=os.getenv("SESSION_SERVICE_URI", "pgclean://localhost"),
    memory_service_uri=os.getenv("MEMORY_SERVICE_URI", "pgmemory://localhost"),
    allow_origins=["*"],
    web=True,
)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
    )
