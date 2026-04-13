import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the catalog of Song objects to be ranked."""
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        """Return a numeric score for one song against a UserProfile (max 4.0)."""
        score = 0.0
        if song.genre == user.favorite_genre:
            score += 1.0                                          # halved: was 2.0
        if song.mood == user.favorite_mood:
            score += 1.0
        score += 2.0 * (1.0 - abs(song.energy - user.target_energy))  # doubled: was ×1
        return score

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k Song objects ranked by score for the given UserProfile."""
        ranked = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)
        return ranked[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Build a human-readable string listing why song was recommended to user."""
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre match ({song.genre}, +2.0)")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood match ({song.mood}, +1.0)")
        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        reasons.append(f"energy similarity {energy_sim:.2f} (song={song.energy}, target={user.target_energy})")
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song against user preferences.

    Returns:
        score   -- numeric total (max 4.0)
        reasons -- list of strings explaining each contribution, e.g.
                   ["genre match (pop, +2.0)", "energy similarity 0.98"]
    """
    score = 0.0
    reasons = []

    # +1.0 for an exact genre match (halved from 2.0 — energy is now the dominant signal)
    if song["genre"] == user_prefs.get("genre", ""):
        score += 1.0
        reasons.append(f"genre match ({song['genre']}, +1.0)")

    # +1.0 for an exact mood match
    if song["mood"] == user_prefs.get("mood", ""):
        score += 1.0
        reasons.append(f"mood match ({song['mood']}, +1.0)")

    # 0.0–2.0 based on how close the song's energy is to the user's target
    # doubled weight so energy proximity is now the strongest single signal
    # formula: 2 * (1 - |song.energy - target_energy|)
    energy_sim = 2.0 * (1.0 - abs(song["energy"] - user_prefs.get("energy", 0.5)))
    score += energy_sim
    reasons.append(
        f"energy similarity {energy_sim:.2f} "
        f"(song={song['energy']}, target={user_prefs.get('energy', 0.5)})"
    )

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, List[str]]]:
    """
    Score every song, then return the top-k sorted highest to lowest.

    Uses sorted() instead of list.sort() because sorted() returns a new list
    and leaves the original catalog unchanged — the Pythonic choice when you
    need a ranked copy without mutating the source data.
    """
    scored = [
        (song, *score_song(user_prefs, song))   # (song, score, reasons)
        for song in songs
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
