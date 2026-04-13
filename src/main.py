"""
Command line runner for the Music Recommender Simulation.

Runs multiple user profiles — standard and adversarial — to evaluate how
the scoring logic behaves across different and edge-case preference sets.
"""

from .recommender import load_songs, recommend_songs


# ---------------------------------------------------------------------------
# User profiles
# ---------------------------------------------------------------------------

PROFILES = {
    # --- Standard profiles ---
    "High-Energy Pop": {
        "genre": "pop", "mood": "happy", "energy": 0.9
    },
    "Chill Lofi": {
        "genre": "lofi", "mood": "chill", "energy": 0.35
    },
    "Deep Intense Rock": {
        "genre": "rock", "mood": "intense", "energy": 0.95
    },

    # --- Adversarial / edge-case profiles ---

    # Conflicting signals: genre weight should dominate over the mood mismatch
    "Conflicting: High-Energy but Chill Mood": {
        "genre": "ambient", "mood": "chill", "energy": 0.95
    },

    # Genre that doesn't exist in the catalog — pure energy + mood fallback
    "Ghost Genre (country)": {
        "genre": "country", "mood": "happy", "energy": 0.7
    },

    # Extreme low energy — tests whether low-energy songs always win
    "Floor Energy": {
        "genre": "lofi", "mood": "focused", "energy": 0.0
    },
}


# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------

def print_recommendations(label: str, user_prefs: dict, recommendations: list) -> None:
    width = 54
    print(f"\n{'=' * width}")
    print(f"  Profile : {label}")
    print(f"  Prefs   : {user_prefs}")
    print(f"{'=' * width}")
    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"      Score : {score:.2f} / 4.00  [genre×0.5, mood×1, energy×2]")
        for reason in reasons:
            print(f"              • {reason}")
        print(f"  {'-' * (width - 2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for label, prefs in PROFILES.items():
        recs = recommend_songs(prefs, songs, k=5)
        print_recommendations(label, prefs, recs)


if __name__ == "__main__":
    main()
