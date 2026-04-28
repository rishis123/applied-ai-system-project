"""
config.py — Central configuration. All secrets come from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── embedding ──────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, free, ~80MB, runs on CPU

# ── chromadb ───────────────────────────────────────────────────────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "spotify_tracks"

# ── TF-IDF rankings ───────────────────────────────────────────────────────
RANKINGS_PATH = os.getenv("RANKINGS_PATH", "./rankings.pkl")

# ── retrieval ──────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "12"))   # blurbs retrieved per query

# ── gemini ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# GEMINI_API_KEY absence is checked lazily in rag.py when the client is first used,
# so batch/scoring mode can run without a key set.
