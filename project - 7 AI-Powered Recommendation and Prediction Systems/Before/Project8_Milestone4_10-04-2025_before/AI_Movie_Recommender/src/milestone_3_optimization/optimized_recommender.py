# optimized_recommender.py
# =============================================================
# Milestone 3: Optimized Movie Recommender
# This class is a performance-enhanced version of the basic
# recommender. It uses cosine similarity but can be swapped
# with more scalable methods if needed.
# =============================================================

import pandas as pd
from sklearn.neighbors import NearestNeighbors

class OptimizedMovieRecommender:
    """
    An optimized KNN-based movie recommendation system that
    uses cosine similarity with brute-force search. The core
    logic is the same as the basic recommender but built with
    performance optimization in mind.
    """

    def __init__(self, ratings_file):
        """
        Load ratings and initialize the nearest neighbor model.

        Parameters:
            ratings_file (str): Path to CSV file with user-movie ratings.
        """
        self.ratings = pd.read_csv(ratings_file)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')

    def train(self):
        """
        Create the user-movie matrix and fit the similarity model.
        Fills missing ratings with 0 for proper computation.
        """
        user_movie_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.model.fit(user_movie_matrix)
        self.user_movie_matrix = user_movie_matrix

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend top N movies based on similar users' preferences.

        Parameters:
            user_id (int): ID of the user requesting recommendations
            n_recommendations (int): How many recommendations to return

        Returns:
            List of movie IDs most likely to be liked by the user
        """
        user_vector = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_recommendations + 1)

        # Ignore the user themselves
        recommended_users = indices.flatten()[1:]

        # Average the movie preferences of similar users
        recommended_movies = self.user_movie_matrix.iloc[recommended_users].mean().sort_values(ascending=False)
        return recommended_movies.head(n_recommendations).index.tolist()
