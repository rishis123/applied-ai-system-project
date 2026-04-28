## Reflection and Ethics

## AI Reliability & Testing Summary
Primarily determined through human evaluation of AI output - 6/6 tests passed for quality, even with sparse music within a genre/type. The best advice I have is give clear, specific information in your query, to improve the semantic search quality.

## Limitations and Biases
The system is limited by the small dataset from Kaggle, and can be improved by using a Spotify API or more comprehensive dataset.


## Misuse and Prevention
Make sure you don't commit your keys or other sensitive information -- follow standard env and gitignore practice.

## AI Collaboration
During development, AI (primarily Claude Code) acted as a junior engineer/primary programmer while I provided insight, reviewed work, and challenged results.
* **Helpful Suggestion:** It really helped with the UI, which I didn't need to modify much.
* **Flawed Suggestion:** It recommends earlier Gemini flash versions, and struggled to download Kaggle datasets using the API.

##  Conclusion
This project taught me the importance of the information retrieval pipelines backing RAG. The primary difference between this and far worse/better models is the data availability and cleanliness. genAI is a useful helper but is upper-bounded by your prompt quality and how you provide context.