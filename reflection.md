# Reflection: Profile Comparisons

## High-Energy Pop vs. Chill Lofi

Both profiles hit their top slot cleanly — *Sunrise City* (pop/happy) and
*Library Rain* (lofi/chill) — because each has a song in the catalog that
matches all three signals. The interesting difference is in slots 3–5. The
High-Energy Pop profile's lower ranks fill with songs that share *only* energy
proximity (no genre, no mood), while the Chill Lofi profile's lower ranks
still include genre matches. This makes sense: the lofi sub-catalog is bigger
and more tightly clustered in energy (0.35–0.42), so even the third-best lofi
song sits close in energy. Pop songs are more spread out, so energy-only
fallbacks have a wider range. The takeaway: **catalog density within a genre
matters as much as the weights themselves**.

---

## Chill Lofi vs. Deep Intense Rock

The Lofi profile gets a deep, consistent top-3 (all lofi songs). The Rock
profile collapses after #1 because there is only one rock song in the catalog.
Slots 2–5 are filled entirely by energy proximity — *Gym Hero* (pop/intense)
appears at #2 purely because its energy (0.93) is close to the rock target
(0.95). This is the catalog-homogeneity problem in action: **a user asking for
rock gets one rock song and four coincidentally loud songs from unrelated
genres**. Under original weights, mood match gave *Gym Hero* a +1.0 bonus on
top of energy; under revised weights (energy×2) it barely changed rank because
the energy contribution already dominated. The rock user and the lofi user
nominally "use the same algorithm," but the quality of their experience is
completely different because of how the catalog is distributed.

---

## High-Energy Pop vs. Conflicting (ambient/chill/0.95)

These two profiles share the same target energy (≈0.9) but ask for completely
different genre and mood combinations. The Pop profile returns a coherent
playlist because pop + happy + high-energy is well represented. The Conflicting
profile exposes a real scoring flaw: *Spacewalk Thoughts* (ambient/chill,
energy 0.28) ranks #1 because its genre+mood bonus (+2.0 in old weights,
+2.0 in new) outweighs a massive energy penalty of −1.34 (old) or −1.34×2
(new). Under old weights the gap was wide enough that no energy-close song
could catch it. Under new weights (energy×2) the gap narrowed to where
*Gym Hero* (score 1.96) nearly tied it. **The weight shift made the conflicting
profile more honest** — it no longer pretends a 0.28-energy song is a great
match for a 0.95-energy target. But it still ranks #1, which would likely
frustrate a real user who wanted something high-energy to relax to.

---

## Ghost Genre (country) vs. any genre-present profile

The country profile's max reachable score is ~2.9, while a pop user's max is
4.0. This 1-point ceiling difference is the direct result of the genre bonus
never firing. What is striking is that the **output looks the same format** as
a normal recommendation — there is no indicator that the genre was unmatched.
A real user would have no idea their experience was degraded. Comparing against
the Lofi profile (which hits 4.0) makes the gap stark: the country user's best
result scores lower than the lofi user's third-best result. This comparison
shows that **missing catalog coverage is invisible to the user but very visible
in the scores**.

---

## Floor Energy (lofi/focused/0.0) — old weights vs. new weights

Under old weights (energy×1), *Focus Flow* scored 3.60 and the lofi genre
anchor dominated. Under new weights (energy×2), *Focus Flow* dropped to 3.20
because its energy (0.40) is penalized more heavily against a target of 0.0.
The lofi non-focused songs (*Library Rain*, *Midnight Coding*) also dropped.
The most energy-close songs in the catalog (*Glacier Blue* at 0.20, *Bamboo
Wind* at 0.25) moved up relative to the lofi cluster. This comparison shows
that **doubling the energy weight does not change who ranks #1 when genre+mood
is strong, but it does shift the order of everyone below** — exactly the
behavior you would expect from reweighting a tiebreaker signal.
