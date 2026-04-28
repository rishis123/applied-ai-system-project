"""
Command-line entry point for the Wavefinder Music Recommender.

Default (interactive RAG mode):
  python -m cli.main

Batch scoring mode (no API key required):
  python -m cli.main --batch

The interactive mode uses the high_popularity_spotify_data.csv dataset,
ingests it into ChromaDB on first run, then enters a loop where each
natural-language query triggers the full RAG pipeline: semantic retrieval
from ChromaDB → Gemini generation → grounded recommendation displayed
in the terminal.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SPOTIFY_CSV = "data/high_popularity_spotify_data.csv"


# ── RAG setup ─────────────────────────────────────────────────────────────

def _ensure_db() -> bool:
    """
    Ingest high_popularity_spotify_data.csv into ChromaDB if not already built.
    Returns True on success, False on failure.
    """
    from rag.config import CHROMA_PATH, RANKINGS_PATH
    if Path(CHROMA_PATH).exists() and Path(RANKINGS_PATH).exists():
        logger.info("ChromaDB and rankings found — skipping ingest.")
        return True
    logger.info("Database not found — running first-time ingest from %s …", SPOTIFY_CSV)
    try:
        from rag.ingest import ingest
        ingest(SPOTIFY_CSV)
        return True
    except FileNotFoundError:
        logger.error("Song catalog not found: %s", SPOTIFY_CSV)
        return False
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        return False


# ── modes ──────────────────────────────────────────────────────────────────

def interactive_mode() -> None:
    """RAG-powered interactive recommendation loop."""
    if not _ensure_db():
        print(
            "ERROR: Could not build the vector database. "
            "Check that data/high_popularity_spotify_data.csv exists and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from rag.rag import query_recommender
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("\n╔══════════════════════════════════════════╗")
    print("║  Wavefinder AI — RAG Music Recommender  ║")
    print("╚══════════════════════════════════════════╝")
    print('Type a music question and press Enter. Type "quit" to exit.\n')
    print("Examples:")
    print('  "I want something chill and acoustic for studying"')
    print('  "Recommend high-energy electronic tracks"')
    print('  "What should I listen to if I like jazz but want something upbeat?"\n')

    while True:
        try:
            query = input("What are you in the mood for? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        logger.info("Query: %r", query)
        try:
            result = query_recommender(query)
        except ValueError as exc:
            print(f"Invalid input: {exc}\n")
            continue
        except RuntimeError as exc:
            logger.error("RAG pipeline error: %s", exc)
            print(f"Error: {exc}\n")
            continue
        except Exception as exc:
            logger.error("Unexpected error during query: %s", exc)
            print(f"Something went wrong: {exc}\n")
            continue

        print(f"\n{result['answer']}")

        tracks = result.get("tracks", [])
        if tracks:
            print(f"\n── Retrieved from catalog ({result.get('retrieval_mode', '?')} mode) ──")
            for t in tracks[:5]:
                name     = t.get("track_name", "Unknown")
                artist   = t.get("track_artist", "Unknown")
                genre    = t.get("playlist_genre", "")
                subgenre = t.get("playlist_subgenre", "")
                score    = t.get("similarity_score")
                detail   = " · ".join(filter(None, [genre, subgenre]))
                score_str = f"  {score:.3f}" if score is not None else ""
                print(f"  • {name} by {artist}" +
                      (f"  [{detail}]" if detail else "") +
                      score_str)
        print()


def batch_mode() -> None:
    """
    Score built-in profiles against the dataset using TF-IDF retrieval.
    No Gemini API key required — retrieval only.
    """
    if not _ensure_db():
        print("ERROR: Could not build the database.", file=sys.stderr)
        sys.exit(1)

    # FIX: retrieve_by_rankings now returns (docs, metas, scores) — 3-tuple
    from rag.rag import retrieve_by_rankings

    profiles = {
        "High-Energy Pop":                         {"genre": "pop",        "valence": 0.8,  "energy": 0.9},
        "Chill Lofi":                              {"genre": "lofi",       "valence": 0.5,  "energy": 0.35},
        "Deep Intense Rock":                       {"genre": "rock",       "valence": 0.4,  "energy": 0.95},
        "Conflicting: High-Energy but Chill Mood": {"genre": "electronic", "valence": 0.5,  "energy": 0.95},
        "Ghost Genre (country)":                   {"genre": "country",    "valence": 0.8,  "energy": 0.7},
        "Floor Energy":                            {"genre": "electronic", "valence": 0.6,  "energy": 0.9},
    }

    for label, prefs in profiles.items():
        query = (
            f"{prefs['genre']} music with energy {prefs['energy']} "
            f"and valence {prefs['valence']}"
        )
        print(f"\n{'=' * 54}")
        print(f"  Profile : {label}")
        print(f"  Prefs   : {prefs}")
        print(f"  Query   : {query}")
        print(f"{'=' * 54}")

        try:
            # FIX: unpack 3-tuple (docs, metas, scores) — was (track_id, meta)
            docs, metas, scores = retrieve_by_rankings(query, top_k=5)
        except Exception as exc:
            logger.error("Retrieval failed for profile '%s': %s", label, exc)
            print(f"  ERROR: {exc}")
            continue

        if not metas:
            print("  No results found.")
        else:
            for rank, (meta, score) in enumerate(zip(metas, scores), start=1):
                name   = meta.get("track_name", "Unknown")
                artist = meta.get("track_artist", "Unknown")
                genre  = meta.get("playlist_genre", "")
                energy = meta.get("energy", "?")
                print(f"  #{rank}  {name} by {artist} [{genre}]  score={score:.4f}  energy={energy}")

        print(f"  {'-' * 52}")


# ── entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wavefinder Music Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Interactive mode (default): RAG + Gemini — requires GEMINI_API_KEY in .env\n"
            "Batch mode (--batch):       local scoring only, no API key needed"
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch scoring evaluation against all built-in profiles",
    )
    args = parser.parse_args()

    if args.batch:
        batch_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()