# WaveFinder: Music Recommender driven by RAG and Semantic-Search 

## Project Context
**WaveFinder** was originally developed as a CLI tool, with limited music options, that recommends music based on proximity to several numerical fields, including energy and tempo. It was very limited in capacity and scope, and lacked insight based on genre and semantics.

---
## Loom Link
https://www.loom.com/share/837b4819f0db4fb288351536277cd294

## Title and Summary
Wavefinder is an intelligent, RAG-powered (Retrieval-Augmented Generation) music recommendation engine that provides grounded, conversational suggestions based on a user’s mood, vibe, or specific metadata. Unlike standard LLMs that may hallucinate non-existent songs, Wavefinder uses a hybrid retrieval pipeline—combining semantic vector search for abstract "vibes" with TF-IDF rankings for popularity and chart-based queries—to ensure every recommendation is backed by real-world data.

This project matters because it solves the "grounding problem" in AI-driven discovery; by forcing the LLM to cite a specific, retrieved catalog, the system provides trustworthy recommendations that align perfectly with the user's specific audio preferences and the actual constraints of the music library.

Key Technical Features (System design in /assets summary):

Hybrid Retrieval Pipeline: Uses a heuristic router to automatically switch between semantic search (ChromaDB) and popularity-weighted keyword search (TF-IDF) based on the user's query.

Contextual Ingestion: Transforms raw Spotify metrics into descriptive natural-language blurbs to bridge the gap between numerical audio data and human-centric queries.

Grounded Generation: Utilizes Google Gemini to synthesize recommendations, restricted by a strict system prompt to only suggest tracks found within the retrieved data.

Citation Guardrails: Includes a post-generation verification step that checks LLM outputs against the retrieved catalog to detect and flag any potential hallucinations.



## Setup Instructions
To run this project locally, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rishis123/applied-ai-system-project.git
    cd applied-ai-system-project
    ```
2.  **Install Dependencies:**
    Ensure you have Python 3.9+ installed.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables:**
    Create a `.env` file in the root directory and add your API key:
    ```env
    GEMINI_API_KEY=your_actual_key_here
    ```
4.  **Launch the System:**
    ```bash
    uvicorn rag.server:app --reload
    ```

## Sample Interactions

### Example 1: 
* **Input:** "High-energy workout tracks"
* **AI Output:** "Here are some high-energy tracks from your catalog that are well-suited for a workout. I've focused on tracks explicitly labeled as high-energy, with genres like electronic, pop, and gaming often providing the driving rhythm needed for exercise. While most have a good tempo for workouts, I've noted any instances where the tempo might be a bit lower than expected for a "high-energy" track.

[6] "Get Up (Rattle) [feat. Far East Movement] - Vocal Edit" by Bingo Players, Far East Movement — This track boasts the highest energy level in the catalog (0.98) and a strong dance beat, making it ideal for high-intensity workouts.
[4] "This Is What It Feels Like" by Armin van Buuren, Trevor Guthrie — A high-energy electronic track with a driving tempo (129.88 BPM) that's great for maintaining a steady pace during exercise.
[2] "Around the World" by Daft Punk — This iconic high-energy electronic track has exceptional danceability (0.96) and a consistent beat perfect for a workout.
[7] "Miles On It" by Marshmello, Kane Brown — A high-energy gaming track with a good tempo (129.96 BPM) and an upbeat vibe to keep you motivated.
[3] "Push Up - Main Edit" by Creeds — While labeled as high-energy (0.83) and from the gaming genre, note that its tempo (75.02 BPM) is lower, which might be better for power-focused or slower, intense movements rather than high-speed cardio."

### Example 2: 
* **Input:** "Chill indie music for a rainy day"
* **AI Output:** "I've selected tracks from the catalog that best capture the "chill indie music for a rainy day" vibe. While strictly "indie" genres are few, I've prioritized songs from artists with indie sensibilities or genres like alternative and ambient that fit the desired mood, focusing on lower energy and melancholic valence, noting where the "chill" aspect might be slightly less prominent due to tempo or energy.

Here are your recommendations:

[4] "Sparks" by Coldplay — This ambient track perfectly embodies the "chill rainy day" mood with its low energy and melancholic tone, from an artist often associated with indie sensibilities.
[9] "Paradise" by Coldplay — With a mid-energy level and melancholic valence, this soft pop track offers a reflective atmosphere well-suited for a rainy day, from an indie-adjacent artist.
[5] "Sweater Weather" by The Neighbourhood — While more energetic than strictly "chill," this track is a strong indie-pop genre match with a melancholic undertone that fits the rainy day mood.
[1] "S*x, Drugs, Etc." by Beach Weather — This alternative rock track provides a strong "indie" genre match, though its higher energy might be less "chill" than your request, it still offers a compelling mood."

### Example 3: 
* **Input:** "Happy songs with high valence"
* **AI Output:** "I've identified several tracks from the catalog that perfectly match your request for happy songs with high valence. These recommendations are all described as "happy and upbeat" based on their valence scores, ensuring they convey a positive mood.

Here are your recommendations:

[7] "Best of My Love" by The Emotions — This classic pop track boasts the highest valence, making it a truly upbeat and happy choice.
[5] "CUTE DEPRESSED" by Dyan Dxddy — Despite the name, this gaming track has a very high valence and is described as happy and upbeat.
[9] "Taste" by Sabrina Carpenter — A high-energy pop track with a strong happy and upbeat valence score.
[6] "Mood (feat. iann dior)" by 24kGoldn, iann dior — This gaming track is high-energy and clearly categorized as happy and upbeat.
[8] "Taste" by Sabrina Carpenter — Another high-energy pop track from Sabrina Carpenter, also rated as happy and upbeat.
[4] "Bed Chem" by Sabrina Carpenter — A high-energy pop track from Sabrina Carpenter, noted for its happy and upbeat valence.
[1] "Happier (feat. Clementine Douglas)" by The Blessed Madonna, Clementine Douglas — This electronic track is high-energy and explicitly labeled as happy and upbeat."



## Design Decisions
1. Natural Language Data Synthesis (The "Blurb" Strategy)
Rather than simply storing raw numerical data (e.g., energy: 0.8) in the vector database, the system converts track metadata into descriptive natural-language sentences.

Why: This bridges the gap between how users describe music (using adjectives like "high-energy" or "melancholic") and how the data is stored.

Implementation: The track_blurb function uses conditional logic to translate raw scores into human descriptors, such as mapping a high valence score to "happy and upbeat".

2. Hybrid Retrieval Pipeline
The system does not rely on a single search method; it uses a heuristic router to choose the best tool for the job.

Semantic Search (ChromaDB): Used for abstract, mood-based queries like "something chill for studying".

TF-IDF + Popularity Ranking: Used when users ask for "popular," "top," or "trending" tracks.

Trade-off: This adds complexity to the code but prevents semantic search from missing literal keyword matches or popularity signals that a pure vector search might overlook.

3. Weighted Popularity Scoring
In the TF-IDF retrieval mode, the system calculates a combined score rather than relying solely on keyword relevance.

Formula: The score is calculated as 0.7×tfidf_cosine_similarity+0.3×normalized_popularity.

Impact: This ensures that if multiple songs match a keyword, the system prioritizes the one that is more likely to be a "hit," reflecting real-world user preferences.

4. Hallucination Guardrails & Grounding
To prevent the LLM from suggesting songs that don't exist in the local dataset, the system implements a "closed-loop" logic.

Numbered Cataloging: Retrieved tracks are formatted into a numbered list before being sent to the LLM, forcing the model to select from a specific set of options.

Post-Generation Verification: The _verify_response function scans the AI's output for citations. If the AI references a number outside the retrieved set (e.g., citing track "[11]" when only 10 were provided), it flags the error to the user.

5. Robust Data Ingestion & Error Handling
The ingestion process includes specific "defensive" coding decisions to handle real-world CSV data inconsistencies.

Safe Conversions: Functions like safe_cmp and safe_display prevent the system from crashing if technical audio metrics are missing or improperly formatted in the CSV.

Collision Prevention: The system explicitly renames the CSV's type column to spotify_type to avoid metadata conflicts with internal ChromaDB keys.

6. Evaluation Design (Batch Mode)
The project includes a dedicated --batch mode that allows for testing without an active LLM connection.

Purpose: This enables you to test the retrieval logic and scoring accuracy against "built-in profiles" (like "High-Energy Pop" or "Chill Lofi") without spending API credits on Gemini.

Learning: This separation of concerns allows for debugging the retrieval (finding the right data) independently from the generation (explaining the data).

## Reliability and Evaluation
To ensure the system provides accurate musical data, I implemented the following:

### Automated Validation: 
A unit test script verifies that every track recommended by the AI exists in the local database.

### Citation Guardrails: 
A verification function that scans for out-of-range citations to prevent the LLM from suggesting "ghost" tracks.


