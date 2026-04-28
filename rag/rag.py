"""
rag.py — Retrieval-Augmented Generation query engine.

Pipeline:
  1. Query is embedded and matched against ChromaDB (semantic) or
     TF-IDF + popularity index (keyword/popularity).
  2. Retrieved track blurbs are formatted into a numbered catalog.
  3. Gemini is instructed to recommend from the numbered catalog,
     always picking the best available matches even if imperfect.
  4. Response is returned with retrieved tracks + per-track similarity scores.

Run ingest first:
  python -m rag.ingest --csv data/high_popularity_spotify_data.csv
"""

import logging
import pickle
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

from .config import (
    CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL, RANKINGS_PATH, TOP_K,
)

logger = logging.getLogger(__name__)

# ── singletons ─────────────────────────────────────────────────────────────

_embed_model: SentenceTransformer | None = None
_chroma_client = None
_collection = None
_gemini_client: genai.Client | None = None
_rankings: dict | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_collection():
    global _collection, _chroma_client
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            _collection = _chroma_client.get_collection(COLLECTION_NAME)
            logger.info(
                "Connected to ChromaDB collection '%s' (%d docs).",
                COLLECTION_NAME,
                _collection.count(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found at '{CHROMA_PATH}'. "
                "Run the ingest step first:\n"
                "  python -m rag.ingest --csv data/high_popularity_spotify_data.csv"
            ) from exc
    return _collection


def _get_rankings() -> dict:
    global _rankings
    if _rankings is None:
        rankings_path = Path(RANKINGS_PATH)
        if not rankings_path.exists():
            raise RuntimeError(
                f"Rankings file not found at '{RANKINGS_PATH}'. "
                "Run the ingest step first:\n"
                "  python -m rag.ingest --csv data/high_popularity_spotify_data.csv"
            )
        with open(rankings_path, 'rb') as f:
            _rankings = pickle.load(f)
        logger.info("Loaded TF-IDF rankings (%d tracks).", len(_rankings['track_ids']))
    return _rankings


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialised (model: %s).", GEMINI_MODEL)
    return _gemini_client


# ── retrieval ──────────────────────────────────────────────────────────────

def _is_track(meta: dict) -> bool:
    return meta.get("doc_type") == "track"


def retrieve_semantic(query: str, top_k: int = TOP_K) -> tuple[list[str], list[dict], list[float]]:
    """
    Embed the query and retrieve the top-k semantically similar track blurbs.
    Returns (docs, metas, similarity_scores). Scores are cosine similarities
    computed from the raw ChromaDB distance (distance = 1 - cosine_similarity).
    Artist docs are filtered out; fetches 2x candidates to guarantee top_k tracks.
    """
    logger.info("Semantic retrieval for query: %r", query[:80])
    model = _get_embed_model()
    collection = _get_collection()
    query_vec = model.encode([query]).tolist()

    candidates = top_k * 2
    results = collection.query(
        query_embeddings=query_vec,
        n_results=candidates,
        include=["documents", "metadatas", "distances"],
    )
    docs_raw = results["documents"][0]
    metas_raw = results["metadatas"][0]
    # ChromaDB cosine space returns distances where distance = 1 - similarity
    distances_raw = results["distances"][0]

    track_docs, track_metas, track_scores = [], [], []
    for doc, meta, dist in zip(docs_raw, metas_raw, distances_raw):
        if _is_track(meta):
            track_docs.append(doc)
            track_metas.append(meta)
            track_scores.append(round(1.0 - dist, 4))  # convert distance → similarity
        if len(track_docs) == top_k:
            break

    logger.info(
        "Semantic retrieval: %d candidates -> %d tracks. "
        "Score range: %.3f – %.3f",
        len(docs_raw), len(track_docs),
        min(track_scores, default=0), max(track_scores, default=0),
    )
    return track_docs, track_metas, track_scores


def retrieve_by_rankings(query: str, top_k: int = TOP_K) -> tuple[list[str], list[dict], list[float]]:
    """
    Retrieve top tracks using TF-IDF similarity + popularity weighting.
    Returns (docs, metas, combined_scores).
    combined_score = 0.7 * tfidf_cosine + 0.3 * normalised_popularity
    """
    logger.info("TF-IDF retrieval for query: %r", query[:80])
    rankings = _get_rankings()
    vectorizer = rankings['vectorizer']
    tfidf_matrix = rankings['tfidf_matrix']
    track_ids = rankings['track_ids']
    popularity = rankings['popularity']

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    pop_normalized = np.array(popularity) / 100.0
    combined_scores = 0.7 * similarities + 0.3 * pop_normalized

    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    docs, metas, scores = [], [], []
    collection = _get_collection()
    for idx in top_indices:
        track_id = track_ids[idx]
        try:
            result = collection.get(ids=[track_id], include=["documents", "metadatas"])
            if result["metadatas"] and _is_track(result["metadatas"][0]):
                docs.append(result["documents"][0])
                metas.append(result["metadatas"][0])
                scores.append(round(float(combined_scores[idx]), 4))
        except Exception as e:
            logger.warning("Could not fetch track %s from ChromaDB: %s", track_id, e)

    logger.info("TF-IDF retrieval: %d tracks returned.", len(docs))
    return docs, metas, scores


def _select_retrieval_mode(query: str) -> bool:
    """
    Heuristic: route popularity/chart queries to TF-IDF,
    mood/vibe/descriptor queries to semantic search.
    Returns True to use TF-IDF rankings.
    """
    tfidf_signals = ["popular", "top", "best", "trending", "most played", "chart"]
    return any(signal in query.lower() for signal in tfidf_signals)


# ── generation ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a music recommender assistant.
You will receive a numbered catalog of tracks retrieved from a music database.
Your job is to recommend the best matching tracks from this catalog for the user's request.

Rules:
- ALWAYS recommend tracks — pick the closest matches even if the fit is imperfect.
- Never refuse or say you cannot find matches. Instead, recommend the best available
  tracks and briefly note how they relate to (or differ from) the request.
- Reference each recommendation by catalog number, track name, and artist.
- Format each recommendation as: [N] "Track Name" by Artist Name — one-line reason.
- Do not invent or hallucinate tracks outside the provided catalog.
- Begin with one short paragraph explaining your rationale and any caveats about fit."""


def _build_catalog(docs: list[str], metas: list[dict], scores: list[float]) -> str:
    """
    Format retrieved tracks as a numbered catalog for the LLM prompt.
    Includes similarity score so Gemini can reason about match quality.
    """
    lines = []
    for i, (doc, meta, score) in enumerate(zip(docs, metas, scores), start=1):
        name = meta.get("track_name", "Unknown")
        artist = meta.get("track_artist", "Unknown")
        genre = meta.get("playlist_genre", "")
        subgenre = meta.get("playlist_subgenre", "")
        energy = meta.get("energy", "N/A")
        tempo = meta.get("tempo", "N/A")
        lines.append(
            f"[{i}] \"{name}\" by {artist}\n"
            f"    Genre: {genre} / {subgenre} | Energy: {energy} | "
            f"Tempo: {tempo} BPM | Similarity: {score:.3f}\n"
            f"    {doc}"
        )
    return "\n\n".join(lines)


def _verify_response(response_text: str, metas: list[dict]) -> list[int]:
    """
    Post-generation guard: check that all [N] citations in Gemini's response
    fall within the retrieved catalog range. Logs warnings for out-of-range refs.
    """
    cited = {int(n) for n in re.findall(r'\[(\d+)\]', response_text)}
    valid_range = set(range(1, len(metas) + 1))
    hallucinated = sorted(cited - valid_range)
    if hallucinated:
        logger.warning(
            "Gemini cited catalog numbers outside retrieved set: %s (valid: 1-%d)",
            hallucinated, len(metas)
        )
    return hallucinated


def generate(query: str, docs: list[str], metas: list[dict], scores: list[float]) -> str:
    """
    Build a numbered catalog from retrieved tracks and pass it to Gemini.
    The model must cite tracks by number, grounding its output in the
    retrieved set rather than its training knowledge.
    """
    if not docs:
        raise ValueError("No documents retrieved — cannot generate a grounded response.")

    catalog = _build_catalog(docs, metas, scores)
    prompt = (
        f"RETRIEVED TRACK CATALOG ({len(docs)} tracks):\n\n"
        f"{catalog}\n\n"
        f"USER REQUEST: {query}\n\n"
        "Recommend the best matching tracks from the catalog above. "
        "Always provide recommendations — note any fit caveats rather than refusing."
    )

    logger.info(
        "Sending %d tracks to %s. Prompt: %d chars.",
        len(docs), GEMINI_MODEL, len(prompt)
    )

    try:
        client = _get_gemini()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        answer = response.text
        logger.info("Generation complete (%d chars).", len(answer))

        bad_refs = _verify_response(answer, metas)
        if bad_refs:
            answer += (
                f"\n\n⚠️ Note: Response referenced catalog entries {bad_refs} "
                "which are outside the retrieved set."
            )
        return answer

    except Exception as exc:
        logger.error("Gemini generation failed: %s", exc)
        raise


# ── public API ─────────────────────────────────────────────────────────────

def query_recommender(question: str, use_rankings: bool | None = None) -> dict:
    """
    Full RAG pipeline:
      1. Validate input.
      2. Auto-select or use override for retrieval mode.
      3. Retrieve top-k tracks with similarity scores.
      4. Build numbered catalog and send to Gemini with grounding instructions.
      5. Verify citations, return answer + full track detail including scores.

    Args:
        question:     The user's music recommendation request.
        use_rankings: None = auto (heuristic). True = TF-IDF. False = semantic.

    Returns:
        {
          "answer": str,
          "tracks": list[dict],   # includes energy, tempo, danceability, valence,
                                  # loudness, genre, subgenre, similarity_score
          "retrieval_mode": str,  # "semantic" or "tfidf"
        }

    Raises:
        ValueError:       Empty question.
        RuntimeError:     ChromaDB or rankings not built.
        EnvironmentError: GEMINI_API_KEY missing.
    """
    question = question.strip()
    if not question:
        raise ValueError("Question must not be empty.")

    if use_rankings is None:
        use_rankings = _select_retrieval_mode(question)
        logger.info(
            "Auto-selected retrieval mode: %s",
            "tfidf" if use_rankings else "semantic"
        )

    retrieval_mode = "tfidf" if use_rankings else "semantic"

    if use_rankings:
        docs, metas, scores = retrieve_by_rankings(question)
    else:
        docs, metas, scores = retrieve_semantic(question)

    if not docs:
        logger.warning("No tracks retrieved for query: %r", question)
        return {
            "answer": "I couldn't find any tracks in the catalog for that query.",
            "tracks": [],
            "retrieval_mode": retrieval_mode,
        }

    answer = generate(question, docs, metas, scores)

    # Return full track detail for frontend display
    tracks = [
        {
            "track_name": m.get("track_name", "Unknown"),
            "track_artist": m.get("track_artist", "Unknown"),
            "playlist_genre": m.get("playlist_genre", ""),
            "playlist_subgenre": m.get("playlist_subgenre", ""),
            "track_popularity": m.get("track_popularity", 0),
            "energy": m.get("energy", 0),
            "danceability": m.get("danceability", 0),
            "valence": m.get("valence", 0),
            "tempo": m.get("tempo", 0),
            "loudness": m.get("loudness", 0),
            "similarity_score": score,
        }
        for m, score in zip(metas, scores)
    ]

    logger.info(
        "query_recommender complete. Mode: %s. Tracks: %d.",
        retrieval_mode, len(tracks)
    )
    return {
        "answer": answer,
        "tracks": tracks,
        "retrieval_mode": retrieval_mode,
    }