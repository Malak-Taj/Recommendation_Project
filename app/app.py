import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "metadata"       # inside app/
DATA_PATH = BASE_DIR.parent / "data"        # outside app/

# -------------------------------
# Load metadata (pickles)
# -------------------------------
@st.cache_data
def load_metadata():
    movies_content = pd.read_pickle(METADATA_PATH / "movies_content.pkl")
    tfidf_matrix = joblib.load(METADATA_PATH / "tfidf_matrix.pkl")  # use joblib for matrices
    return movies_content, tfidf_matrix

movies_content, tfidf_matrix = load_metadata()

# -------------------------------
# Movie card UI function
# -------------------------------
def movie_card(title, genres, score):
    with st.container():
        st.subheader(title)
        st.caption(f"🎭 Genres: {genres}")
        st.progress(min(score, 1.0))
        st.caption(f"Similarity: {score*100:.0f}%")

# -------------------------------
# Recommendation logic
# -------------------------------
def recommend(movie_title, top_n=10):
    idx = movies_content[movies_content["title"] == movie_title].index[0]

    # cosine similarity using sparse dot product
    similarity_scores = tfidf_matrix.dot(tfidf_matrix[idx].T).toarray().flatten()

    sim_scores = list(enumerate(similarity_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]

    return indices, scores

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("")

selected_movie = st.selectbox(
    "Choose a movie you like:",
    movies_content["title"].values
)

if st.button("Click To Find Similar Movies"):
    with st.spinner("Finding similar movies..."):
        indices, scores = recommend(selected_movie)

    st.subheader("Recommended Movies!")

    # display in 2 rows of 5
    cols = st.columns(5)
    for i in range(5):
        row = movies_content.iloc[indices[i]]
        with cols[i]:
            movie_card(
                title=row["title"],
                genres=row.get("genres", "Unknown"),
                score=scores[i]
            )

    cols = st.columns(5)
    for i in range(5, 10):
        row = movies_content.iloc[indices[i]]
        with cols[i - 5]:
            movie_card(
                title=row["title"],
                genres=row.get("genres", "Unknown"),
                score=scores[i]
            )

else:
    st.info("🔎 Please select a movie and click the button to get recommendations.")
