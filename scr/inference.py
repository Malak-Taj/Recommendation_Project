import joblib
import pandas as pd

# Load ALS model
model = joblib.load("metadata/als_model.pkl")

# Load user-item matrix
user_item_matrix = joblib.load("metadata/user_item_matrix.pkl")

# Load movies content
movies_content = pd.read_parquet("metadata/movies_content.parquet")

# Load TF-IDF matrix
tfidf_matrix = joblib.load("metadata/tfidf_matrix.pkl")

# Load mapping dictionaries
user_to_index = joblib.load("metadata/user_to_index.pkl")
index_to_movie = joblib.load("metadata/index_to_movie.pkl")

from scr.preprocessing_train import hybrid_recommend
def get_recommendations(userId, N=5, alpha=0.5):
    recommendations = hybrid_recommend(
        userId=userId,
        N=N,
        als_model=model,
        user_item_matrix=user_item_matrix,
        movies_content=movies_content,
        user_to_index=user_to_index,
        index_to_movie=index_to_movie,
        tfidf_matrix=tfidf_matrix,
        alpha=alpha
    )
    return recommendations