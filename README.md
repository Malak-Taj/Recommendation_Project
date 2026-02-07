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

Recommendation_Project/
│
├── app/
│   └── app.py                      # Streamlit application (UI)
│
├── metadata/
│   ├── als_model.pkl               # Trained ALS model
│   ├── user_item_matrix.pkl        # Sparse user–item matrix
│   ├── tfidf_matrix.pkl            # TF-IDF feature matrix
│   ├── movies_content.parquet      # Movie metadata (title, genres, content)
│   ├── user_to_index.pkl           # Mapping: userId → ALS index
│   └── index_to_movie.pkl          # Mapping: ALS index → movieId
│
├── scr/
│   └── preprocessing_train.py      # Hybrid recommendation logic
│
├── inference.py                    # Model loading & inference wrapper
├── evaluate.py                     # Simple evaluation / sanity check
│
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation




