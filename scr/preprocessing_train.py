from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def hybrid_recommend(userId, N=5, als_model=None, user_item_matrix=None, 
                     movies_content=None, user_to_index=None, index_to_movie=None, tfidf_matrix=None, alpha=0.5):
    """
    Parameters:
    - userId: original MovieLens userId
    - N: number of recommendations to return
    - als_model: trained ALS model
    - user_item_matrix: sparse user-item matrix for ALS
    - movies_content: DataFrame with 'movieId', 'title', 'genres', 'content'
    - user_to_index: dict mapping original userId -> filtered matrix index
    - index_to_movie: dict mapping ALS index -> original movieId
    - tfidf_matrix: TF-IDF matrix for movie content
    - alpha: weight for content-based similarity
    Returns:
    - DataFrame of top N recommended movies
    """

    #Map userId to row index in filtered matrix
    if userId not in user_to_index:
        raise ValueError(f"userId {userId} not found in filtered dataset")
    user_idx = user_to_index[userId]

    #Get top ALS recommendations
    als_items, als_scores = als_model.recommend(user_idx, user_item_matrix[user_idx], N=20)

    #Map ALS indices to movieId and tf-idf matrix indices
    als_movie_ids = [index_to_movie[i] for i in als_items]
    movie_idx_map = {mid: i for i, mid in enumerate(movies_content['movieId'])}
    als_tfidf_indices = [movie_idx_map[mid] for mid in als_movie_ids if mid in movie_idx_map]

    #Compute content similarity
    similarity_scores = cosine_similarity(tfidf_matrix[als_tfidf_indices], tfidf_matrix)
    content_scores = similarity_scores.sum(axis=0)

    #Combine ALS + content scores
    combined_scores = np.zeros(len(movies_content))
    for i, mid in enumerate(als_movie_ids):
        if mid in movie_idx_map:
            combined_scores[movie_idx_map[mid]] = als_scores[i]
    combined_scores += alpha * content_scores

    #Get top N indices
    top_idx = combined_scores.argsort()[::-1][:N]

    #Return recommended movies
    recommendations = movies_content.iloc[top_idx][['movieId', 'title', 'genres']]
    return recommendations.reset_index(drop=True)
