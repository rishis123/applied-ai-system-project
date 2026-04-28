"""
ingest.py — Run once to build the ChromaDB vector store and TF-IDF rankings.

Usage:
  python -m rag.ingest --csv path/to/spotify.csv

This will:
1. Ingest track data into ChromaDB for semantic search
2. Build TF-IDF rankings for popularity-based retrieval
"""

import argparse
import logging
import math
import pickle
from pathlib import Path

import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CHROMA_PATH, COLLECTION_NAME, EMBED_MODEL, RANKINGS_PATH

logger = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────

def _init_collection(client):
    """Drop and recreate the ChromaDB collection."""
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    return client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def safe(val, decimals=2):
    """Return a rounded float or None for missing values."""
    try:
        f = float(val)
        return round(f, decimals) if not math.isnan(f) else None
    except (TypeError, ValueError):
        return None


def safe_display(val, decimals=2):
    """Return a rounded float string or 'N/A' for display in blurbs."""
    v = safe(val, decimals)
    return str(v) if v is not None else "N/A"


def safe_cmp(val, threshold):
    """Compare a potentially-missing numeric value to a threshold. Missing → False."""
    v = safe(val)
    return v is not None and v > threshold


def track_blurb(row: pd.Series) -> str:
    """Synthesize a natural-language description of a single track."""
    energy = row.get("energy", 0)
    valence = row.get("valence", 0)

    # FIX: use safe_cmp so missing values don't cause TypeError
    energy_desc = (
        "high-energy" if safe_cmp(energy, 0.7)
        else "mid-energy" if safe_cmp(energy, 0.4)
        else "low-energy"
    )
    mood = (
        "happy and upbeat" if safe_cmp(valence, 0.6)
        else "emotionally neutral" if safe_cmp(valence, 0.4)
        else "melancholic or intense"
    )
    return (
        f"{row.get('track_name', 'Unknown')} by {row.get('track_artist', 'Unknown')} "
        f"is a {energy_desc} {row.get('playlist_genre', '')} track "
        f"({row.get('playlist_subgenre', '')}) from the album "
        f"'{row.get('track_album_name', 'Unknown')}' "
        f"released on {row.get('track_album_release_date', 'unknown date')}. "
        f"It has a danceability of {safe_display(row.get('danceability'))}, "
        f"tempo of {safe_display(row.get('tempo'))} BPM, "
        f"loudness of {safe_display(row.get('loudness'))} dB, "
        f"acousticness of {safe_display(row.get('acousticness'))}, "
        f"instrumentalness of {safe_display(row.get('instrumentalness'))}, "
        f"liveness of {safe_display(row.get('liveness'))}, "
        f"speechiness of {safe_display(row.get('speechiness'))}, "
        f"and valence of {safe_display(row.get('valence'))} ({mood}). "
        f"Popularity score: {safe_display(row.get('track_popularity'))}."
    )


def artist_blurbs(df: pd.DataFrame) -> list[tuple[str, str, dict]]:
    """
    Aggregate per-artist stats and return a list of
    (doc_id, blurb_text, metadata) tuples.
    """
    numeric_cols = ["energy", "danceability", "valence", "tempo",
                    "acousticness", "instrumentalness", "speechiness"]
    results = []
    for artist, group in df.groupby("track_artist"):
        genres = group["playlist_genre"].dropna().unique().tolist()
        # FIX: also aggregate subgenres
        subgenres = group["playlist_subgenre"].dropna().unique().tolist()
        stats = {
            col: round(group[col].dropna().mean(), 3)
            for col in numeric_cols
            if col in group.columns
        }
        # FIX: use safe_cmp for missing-safe comparisons
        valence_avg = stats.get("valence", None)
        mood = (
            "upbeat" if (valence_avg is not None and valence_avg > 0.6)
            else "mixed" if (valence_avg is not None and valence_avg > 0.4)
            else "dark/melancholic"
        )
        blurb = (
            f"{artist} appears {len(group)} time(s) in this dataset. "
            f"Their music spans genres: {', '.join(genres)}. "
            f"Subgenres include: {', '.join(subgenres)}. "
            f"On average their tracks have energy {stats.get('energy', 'N/A')}, "
            f"danceability {stats.get('danceability', 'N/A')}, "
            f"valence {stats.get('valence', 'N/A')} ({mood}), "
            f"and tempo {stats.get('tempo', 'N/A')} BPM."
        )
        doc_id = f"artist__{artist[:80]}"
        meta = {"doc_type": "artist", "artist": artist}  # FIX: renamed to avoid collision with CSV 'type' column
        results.append((doc_id, blurb, meta))
    return results


def build_tfidf_rankings(df: pd.DataFrame, track_ids: list[str]) -> dict:
    """
    Build TF-IDF rankings based on track metadata text.
    Returns a dict with tfidf_vectorizer and tfidf_matrix.
    """
    logger.info("Building TF-IDF rankings...")

    texts = []
    for _, row in df.iterrows():
        text = (
            f"{row.get('track_name', '')} "
            f"{row.get('track_artist', '')} "
            f"{row.get('playlist_genre', '')} "
            f"{row.get('playlist_subgenre', '')}"
        )
        texts.append(text)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    rankings = {
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'track_ids': track_ids,
        'popularity': df['track_popularity'].tolist() if 'track_popularity' in df.columns else [0] * len(df)
    }

    Path(RANKINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RANKINGS_PATH, 'wb') as f:
        pickle.dump(rankings, f)

    logger.info(f"TF-IDF rankings saved to {RANKINGS_PATH}")
    return rankings


# ── Spotify CSV ingest ─────────────────────────────────────────────────────

def ingest(csv_path: str, batch_size: int = 500) -> None:
    """Ingest a full Spotify-format CSV into ChromaDB and build TF-IDF rankings."""
    logger.info("Loading CSV from %s …", csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    df = df.drop_duplicates(subset=["track_id"]) if "track_id" in df.columns else df
    logger.info("  %d unique tracks loaded.", len(df))

    # FIX: rename the CSV's 'type' column to avoid collision with ChromaDB metadata key 'type'
    if "type" in df.columns:
        df = df.rename(columns={"type": "spotify_type"})

    logger.info("Loading embedding model …")
    model = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to ChromaDB at %s …", CHROMA_PATH)
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = _init_collection(client)

    # ── track-level documents ──────────────────────────────────────────────
    logger.info("Synthesising track blurbs …")
    track_docs, track_ids, track_metas = [], [], []
    for i, row in df.iterrows():
        blurb = track_blurb(row)
        doc_id = str(row.get("track_id", f"track_{i}"))[:100]
        track_docs.append(blurb)
        track_ids.append(doc_id)
        track_metas.append({
            "doc_type": "track",  # FIX: renamed from 'type' to avoid CSV column collision
            "track_name": str(row.get("track_name", ""))[:100],
            "track_artist": str(row.get("track_artist", ""))[:100],
            "track_album_name": str(row.get("track_album_name", ""))[:100],
            "playlist_genre": str(row.get("playlist_genre", ""))[:50],
            "playlist_subgenre": str(row.get("playlist_subgenre", ""))[:50],
            "track_popularity": safe(row.get("track_popularity")) or 0,
            "energy": safe(row.get("energy")) or 0,
            "danceability": safe(row.get("danceability")) or 0,
            "valence": safe(row.get("valence")) or 0,
            "tempo": safe(row.get("tempo")) or 0,
        })

    # ── artist-level documents ─────────────────────────────────────────────
    logger.info("Synthesising artist blurbs …")
    artist_data = artist_blurbs(df)
    for doc_id, blurb, meta in artist_data:
        track_docs.append(blurb)
        track_ids.append(doc_id)
        track_metas.append(meta)

    # ── embed & upsert in batches ──────────────────────────────────────────
    total = len(track_docs)
    logger.info("Embedding %d documents in batches of %d …", total, batch_size)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        embeddings = model.encode(
            track_docs[start:end], show_progress_bar=False
        ).tolist()
        collection.add(
            documents=track_docs[start:end],
            embeddings=embeddings,
            ids=track_ids[start:end],
            metadatas=track_metas[start:end],
        )
        logger.info("  Upserted %d / %d", end, total)

    logger.info("Ingestion complete.")

    # FIX: pass the full track df and the correct slice of track_ids
    # (not df.head(...) which was doubly wrong: head uses row count, not doc count)
    n_tracks = len(track_ids) - len(artist_data)
    build_tfidf_rankings(df, track_ids[:n_tracks])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Spotify CSV file")
    args = parser.parse_args()
    ingest(args.csv)