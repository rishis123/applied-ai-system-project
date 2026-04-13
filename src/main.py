"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from .recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    # Starter example profile
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print(f"\nTop {len(recommendations)} recommendations for profile {user_prefs}:\n")
    print("-" * 50)
    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        print(f"#{rank}  {song['title']} by {song['artist']}")
        print(f"    Score : {score:.2f} / 4.00")
        print(f"    Why   :")
        for reason in reasons:
            print(f"            • {reason}")
        print("-" * 50)


if __name__ == "__main__":
    main()
