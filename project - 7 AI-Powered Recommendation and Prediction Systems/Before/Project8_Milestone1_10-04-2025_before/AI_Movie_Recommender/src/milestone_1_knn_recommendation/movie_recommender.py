# movie_recommender.py
# =============================================================
# Milestone 1: Basic Movie Recommender using KNN
# TASK: Implement a recommendation system that finds similar
# users and suggests movies based on their ratings.
# =============================================================

import pandas as pd
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    """
    Recommends movies using user-user collaborative filtering
    and cosine similarity via K-Nearest Neighbors.
    """

    def __init__(self, ratings_file):
        """
        Load the ratings data from the specified CSV file.

        Parameters:
            ratings_file (str): Path to the ratings.csv file.
        """
        # Step 1: Load the dataset into self.ratings using pandas
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
        Train the KNN model using the user-movie matrix.
        """
        # Step 3: Use pivot_table to reshape the data into:
        # - Rows: userId
        # - Columns: movieId
        # - Values: rating
        # Fill missing values with 0
        # Store this matrix in self.user_movie_matrix
        # Write your code here
        pass

        # Step 4: Fit the NearestNeighbors model using the matrix
        # Write your code here
        pass

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend movies to the given user based on similar users.

        Parameters:
            user_id (int): The target user ID
            n_recommendations (int): Number of movie IDs to return

        Returns:
            list: Top recommended movie IDs
        """
        # Step 5: Get the user vector from the matrix and reshape it
        # Step 6: Use kneighbors() to find similar users (k=n+1)
        # Step 7: Exclude the current user from neighbors
        # Step 8: Aggregate ratings of neighbors and compute average
        # Step 9: Sort and return top N movie IDs
        # Write your code here
        pass
