# optimized_recommender.py
# =============================================================
# Milestone 3: Optimized Movie Recommender
# TASK: Build a performance-optimized version of the user-based
# recommendation system using cosine similarity.
# =============================================================

import pandas as pd
from sklearn.neighbors import NearestNeighbors

class OptimizedMovieRecommender:
    """
    An optimized movie recommendation system using K-Nearest Neighbors.
    """

    def __init__(self, ratings_file):
        """
        Load and store ratings from the given CSV file.

        Parameters:
            ratings_file (str): Path to user-movie ratings (CSV).
        """
        # Step 1: Load the ratings file using pandas
        # Store it in self.ratings
        # Write your code here
        pass

        # Step 2: Create a NearestNeighbors model with:
        # - metric='cosine'
        # - algorithm='brute'
        # Store it in self.model
        # Write your code here
        pass

    def train(self):
        """
        Create a user-item matrix and fit the nearest neighbor model.
        Missing values should be filled with 0.
        """
        # Step 3: Convert self.ratings into a pivot table:
        # - Rows: userId
        # - Columns: movieId
        # - Values: rating
        # Fill missing values with 0
        # Store it in self.user_movie_matrix
        # Write your code here
        pass

        # Step 4: Fit the NearestNeighbors model to the user-movie matrix
        # Write your code here
        pass

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend top N movies to a given user based on similar users.

        Parameters:
            user_id (int): Target user for recommendations
            n_recommendations (int): Number of recommendations to return

        Returns:
            list of movie IDs
        """
        # Step 5: Retrieve the target user's row from the matrix
        # Step 6: Reshape it and find its nearest neighbors
        # Write your code here
        pass

        # Step 7: Exclude the user from their own neighbors
        # Step 8: Average the ratings of similar users
        # Step 9: Sort and return top N movie IDs
        # Write your code here
        pass
