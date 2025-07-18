# test_recommender.py
# ===========================================================
# Unit tests for Milestone 1: Basic KNN Movie Recommender
# This test verifies that the MovieRecommender class returns
# a list of movie IDs when recommending films for a given user
# based on similarity using the K-Nearest Neighbors algorithm.
# ===========================================================
import unittest
from src.milestone_1_knn_recommendation.movie_recommender import MovieRecommender

class TestMovieRecommender(unittest.TestCase):
    """
    Unit tests for the MovieRecommender class (Milestone 1).
    Verifies that recommendations are generated properly
    based on user similarity using K-Nearest Neighbors (KNN).
    """

    def setUp(self):
        # Initialize and train the recommender using ratings.csv
        self.recommender = MovieRecommender("data/ratings.csv")
        self.recommender.train()

    def test_recommend_returns_list_of_ids(self):
        """
        Test that recommend() returns a list of movie IDs.
        """
        result = self.recommender.recommend(user_id=1, n_recommendations=5)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(isinstance(mid, (int, float)) for mid in result))

if __name__ == "__main__":
    unittest.main()
