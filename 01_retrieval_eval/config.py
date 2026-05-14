"""Configuration loaded from .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).parent
load_dotenv(_HERE / ".env", override=False)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = (_HERE / ".." / "data").resolve()
CHROMA_DIR: Path = (_HERE / "chroma_db").resolve()
RESULTS_DIR: Path = (_HERE / "results").resolve()

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

_raw_models = os.getenv("OLLAMA_EMBED_MODELS", "nomic-embed-text")
OLLAMA_EMBED_MODELS: list[str] = [m.strip() for m in _raw_models.split(",") if m.strip()]

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible)
# ---------------------------------------------------------------------------
LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY: str  = os.getenv("LLM_API_KEY", "")
LLM_MODEL: str    = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Evaluation — k values
# ---------------------------------------------------------------------------
_raw_ks = os.getenv("EVAL_KS", "1,3,5,10")
EVAL_KS: list[int] = [int(k.strip()) for k in _raw_ks.split(",") if k.strip()]

# Primary k used for sorting the report
TOP_K: int = int(os.getenv("TOP_K", "10"))

# ---------------------------------------------------------------------------
# Hybrid — linear alpha sweep
# ---------------------------------------------------------------------------
_raw_alphas = os.getenv("HYBRID_ALPHAS", "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
HYBRID_ALPHAS: list[float] = [float(a.strip()) for a in _raw_alphas.split(",") if a.strip()]

# ---------------------------------------------------------------------------
# Hybrid — RRF sweep
# ---------------------------------------------------------------------------
RRF_K: int = int(os.getenv("RRF_K", "60"))   # standard Cormack RRF constant

_raw_rrf_weights = os.getenv("RRF_WEIGHTS", "0.3,0.5,0.7")
RRF_WEIGHTS: list[float] = [float(w.strip()) for w in _raw_rrf_weights.split(",") if w.strip()]

# ---------------------------------------------------------------------------
# Chain-of-RAG
# ---------------------------------------------------------------------------
CHAIN_TITLE_TOP_N: int   = int(os.getenv("CHAIN_TITLE_TOP_N", "5"))
CHAIN_N_SUBQUESTIONS: int = int(os.getenv("CHAIN_N_SUBQUESTIONS", "3"))
CHAIN_CHUNK_TOP_K: int   = int(os.getenv("CHAIN_CHUNK_TOP_K", "5"))
