# Model Card: VibeFinder 1.0

---

## 1. Model Name

**VibeFinder 1.0**

A content-based music recommender that scores songs against a user's declared
taste profile and returns the best matches with plain-English explanations.

---

## 2. Goal / Task

Given a user's preferred genre, mood, and target energy level, VibeFinder picks
the top 5 songs from a fixed catalog that best match those preferences. It does
not learn from behavior or compare users to each other — it just measures how
close each song is to what the user said they want.

---

## 3. Data Used

- **Size:** 20 songs in `data/songs.csv`
- **Features per song:** title, artist, genre, mood, energy (0.0–1.0),
  tempo_bpm, valence, danceability, acousticness
- **Genres in catalog:** pop, lofi, rock, jazz, synthwave, ambient, soul,
  hip-hop (some appear only once)
- **Moods in catalog:** happy, chill, intense, focused, moody, relaxed, sad
- **Limits:** The dataset is tiny and hand-picked. It does not represent real
  music diversity. Some genres (rock, jazz, ambient) have only one song, so
  users of those genres get a weak experience. The catalog reflects one
  person's taste — not a neutral sample.

---

## 4. Algorithm Summary

Every song in the catalog gets a score out of 4.0. The score has three parts:

1. **Genre match (+1.0):** If the song's genre exactly matches what the user
   said, it gets a bonus. No partial credit — it either matches or it doesn't.

2. **Mood match (+1.0):** Same idea. If the song's mood tag matches the user's
   preferred mood, it earns a point.

3. **Energy proximity (0.0 to 2.0):** The system measures how far apart the
   song's energy is from the user's target energy. A song with the exact same
   energy gets the full 2.0 points. A song on the opposite end of the scale
   gets close to 0. This is calculated as: `2 × (1 − |song energy − target|)`.

After every song is scored, the list is sorted from highest to lowest. The top
5 become the recommendation. Every result also shows which parts of the score
fired and by how much, so the user can see the reasoning.

---

## 5. Observed Behavior / Biases

**Energy can't override a genre+mood lock.**
A song with both a genre and mood match starts with a score floor of 2.0. Even
if its energy is completely wrong, it usually ranks above a song with a
near-perfect energy match but no categorical match. For example: an ambient
song at energy 0.28 still ranked #1 for a user who wanted energy 0.95, simply
because it matched genre and mood. The system *says* it cares about energy, but
in practice it rarely changes who wins at the top.

**Missing genres are invisible.**
If the user asks for a genre that isn't in the catalog (like "country"), the
genre bonus never fires and the system silently degrades. The output looks
exactly the same — no warning, no explanation — but the user's max possible
score drops from 4.0 to around 2.0. A real user would have no idea the system
failed to find their genre at all.

**Catalog gaps create hollow playlists.**
Rock has only one song. So a rock user gets one real genre match and four
"coincidentally loud" songs from pop, hip-hop, or wherever the energy happens
to be close. The playlist looks full but it isn't actually serving the user's
taste past the first result.

**Labels are brittle.**
`"lofi"` and `"lo-fi"` score zero against each other. `"relaxed"` and `"chill"`
score zero against each other. Real listeners know those are basically the same
thing. The system doesn't.

---

## 6. Evaluation Process

Six profiles were tested. Three were standard, three were adversarial (designed
to stress-test the logic):

| Profile | What we looked for |
|---|---|
| `pop / happy / 0.9` | Does it return pop songs? Does *Sunrise City* rank #1? |
| `lofi / chill / 0.35` | Can a song score a perfect 4.0? (*Library Rain* did.) |
| `rock / intense / 0.95` | What happens when only one catalog song matches the genre? |
| `ambient / chill / 0.95` (conflicting) | Does a high-energy user get low-energy songs because of genre match? |
| `country / happy / 0.7` (ghost genre) | What happens when the genre doesn't exist in the catalog? |
| `lofi / focused / 0.0` (floor energy) | Does the system still prefer lofi even when no song is actually at energy 0? |

Each profile was also run under two weight configurations — original (genre×2,
energy×1) and revised (genre×1, energy×2) — to see how the rankings shifted.

The most surprising finding: the conflicting profile showed that categorical
bonuses can absorb almost any numeric penalty. Even after doubling the energy
weight, *Spacewalk Thoughts* (energy 0.28) still ranked #1 for a user who
wanted energy 0.95, because its genre+mood bonus outweighed the energy gap.

---

## 7. Intended Use and Non-Intended Use

**Intended use:**
- Classroom exploration of how content-based filtering works
- Learning how scoring formulas turn data into ranked suggestions
- Experimenting with how weight changes affect recommendations

**Not intended for:**
- Real music discovery for actual listeners
- Any context where recommendation quality or fairness matters
- Replacing tools like Spotify, Last.fm, or any system trained on real
  listening behavior
- Drawing conclusions about music taste at scale — the catalog is 20 songs,
  all hand-picked

---

## 8. Ideas for Improvement

1. **Genre/mood similarity table.** Instead of binary string matching, build a
   small table where `"lofi"` is considered close to `"ambient"` and `"chill"`
   is close to `"relaxed"`. Near-synonym tags would earn partial credit instead
   of zero.

2. **Diversity cap.** Limit how many songs from the same genre can appear in a
   single top-5 result. Right now a lofi user can get 3 lofi songs and 2 filler
   songs. Forcing at least one different genre in the results would make
   recommendations feel less like an echo chamber.

3. **Missing-genre warning.** When the user's preferred genre doesn't appear in
   the catalog at all, surface a message like "No [genre] songs found — showing
   closest matches by mood and energy instead." Silence is the worst UX here.

---

## 9. Personal Reflection

**Biggest learning moment:** I expected
that doubling the energy weight would noticeably change the top results, but for
well-matched profiles (like lofi/chill) almost nothing moved, only adversarial profiles with conflicting signals. 

**How AI tools helped and when I double-checked:** AI-generated code was fast
and mostly correct for boilerplate like CSV loading and list sorting. 

**What surprised me about simple algorithms:** The system had no machine
learning, no training data, and no user history — just three if-statements, but it worked well, showing how it's a good early idea for situations with clear preferences.

**What I would try next:**  I would try adding valence as a signal and ML/NLP components, such as TF-IDF for lyric analysis.
