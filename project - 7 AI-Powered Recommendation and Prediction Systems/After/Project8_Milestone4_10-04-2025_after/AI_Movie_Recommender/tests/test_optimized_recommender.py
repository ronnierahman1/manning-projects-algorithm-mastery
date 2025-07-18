# test_optimized_recommender.py
# ===========================================================
# Unit tests for Milestone 3: Optimized Recommendation System
# This test ensures that the OptimizedMovieRecommender class
# generates valid and efficient movie recommendations using
# optimized neighbor search for scalability.
# ===========================================================
import unittest
from src.milestone_3_optimization.optimized_recommender import OptimizedMovieRecommender

class TestOptimizedRecommender(unittest.TestCase):
    """
    Unit tests for the OptimizedMovieRecommender class (Milestone 3).
    Ensures that optimized recommendation logic works correctly.
    """

    def setUp(self):
        # Load and train the optimized recommender
        self.recommender = OptimizedMovieRecommender("data/ratings.csv")
        self.recommender.train()

    def test_recommendations_length(self):
        """
        Ensure that recommend() returns exactly 5 movie IDs.
        """
        result = self.recommender.recommend(user_id=1, n_recommendations=5)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    def test_recommendation_values_are_ids(self):
        """
        Check that all returned values are valid numeric movie IDs.
        """
        result = self.recommender.recommend(user_id=1, n_recommendations=5)
        self.assertTrue(all(isinstance(mid, (int, float)) for mid in result))

if __name__ == "__main__":
    unittest.main()
