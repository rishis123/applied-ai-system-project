"""
server.py — FastAPI web server for the RAG music recommender.

Usage:
  uvicorn rag.server:app --reload
  Then open http://localhost:8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .rag import query_recommender
from .config import CHROMA_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

STATIC_DIR = Path(__file__).parent / "static"
SPOTIFY_CSV = "data/high_popularity_spotify_data.csv"

# tracks whether startup ingest succeeded — checked on every request
_db_ready: bool = False


def _ensure_db() -> None:
    """
    Ingest high_popularity_spotify_data.csv into ChromaDB if not already built.
    Raises RuntimeError on failure so the lifespan handler can log clearly.
    """
    from .config import RANKINGS_PATH
    if Path(CHROMA_PATH).exists() and Path(RANKINGS_PATH).exists():
        logger.info("ChromaDB and rankings found — skipping ingest.")
        return
    logger.info("Database not found — running first-time ingest from %s …", SPOTIFY_CSV)
    if not Path(SPOTIFY_CSV).exists():
        raise RuntimeError(
            f"CSV not found at '{SPOTIFY_CSV}'. "
            "Place the dataset there or update SPOTIFY_CSV in server.py."
        )
    from .ingest import ingest
    ingest(SPOTIFY_CSV)
    logger.info("Ingest complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_ready
    try:
        _ensure_db()
        _db_ready = True
        logger.info("Server ready.")
    except Exception as exc:
        # Log clearly but don't crash — /health will report not ready
        logger.error("Startup ingest failed: %s", exc)
        _db_ready = False
    yield


app = FastAPI(title="Wavefinder — Music Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── request / response models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    tracks: list[dict]
    retrieval_mode: str  # FIX: was missing — Pydantic was silently stripping it


# ── routes ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/recommend", response_model=QueryResponse)
def recommend(req: QueryRequest):
    # FIX: guard against startup failure
    if not _db_ready:
        raise HTTPException(
            status_code=503,
            detail="Database not ready. Check server logs — ingest may have failed at startup.",
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # FIX: catch and surface errors explicitly instead of letting them become
    # opaque 500s that the frontend shows as empty results
    try:
        result = query_recommender(question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except EnvironmentError as exc:
        # GEMINI_API_KEY missing
        logger.error("Environment error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except RuntimeError as exc:
        # ChromaDB not built, rankings missing, etc.
        logger.error("RAG pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error for query %r: %s", question, exc)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")

    return result


@app.get("/health")
def health():
    """Returns db_ready so you can check whether ingest succeeded at startup."""
    return {"status": "ok", "db_ready": _db_ready}