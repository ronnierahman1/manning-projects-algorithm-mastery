# movie_recommender.py
# =============================================================
# Milestone 1: Basic Movie Recommender using K-Nearest Neighbors
# This class builds a user-based recommendation system using
# cosine similarity to find similar users and recommend movies
# they liked, based on user ratings.
# =============================================================

import pandas as pd
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    """
    A basic movie recommendation system using user-user similarity
    with K-Nearest Neighbors and cosine distance.
    """

    def __init__(self, ratings_file):
        """
        Initialize the recommender by loading the ratings data.

        Parameters:
            ratings_file (str): Path to the CSV file containing user-movie ratings.
        """
        self.ratings = pd.read_csv(ratings_file)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')  # Brute-force search for cosine similarity

    def train(self):
        """
        Train the KNN model using a user-item matrix.
        Missing ratings are filled with 0.
        """
        user_movie_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        self.model.fit(user_movie_matrix)
        self.user_movie_matrix = user_movie_matrix

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend movies to a user based on ratings from similar users.

        Parameters:
            user_id (int): ID of the target user
            n_recommendations (int): Number of movie recommendations to return

        Returns:
            List of movie IDs representing top recommendations
        """
        user_vector = self.user_movie_matrix.loc[user_id].values.reshape(1, -1)
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=n_recommendations + 1)

        # Exclude the user themself from the neighbors
        recommended_users = indices.flatten()[1:]

        # Aggregate ratings of similar users and sort
        recommended_movies = self.user_movie_matrix.iloc[recommended_users].mean().sort_values(ascending=False)
        return recommended_movies.head(n_recommendations).index.tolist()
