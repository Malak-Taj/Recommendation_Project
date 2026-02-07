# Recommendation_Project:
Recommendation Strategy

The system follows a hybrid approach:

ALS Model

Learns user–item interactions

Generates candidate movies per user

Content-Based Similarity

Computes cosine similarity between movie TF-IDF vectors

Enhances semantic relevance (genres, descriptions)

Hybrid Score

Final Score = ALS Score + α × Content Similarity

where α controls the influence of content-based filtering.





