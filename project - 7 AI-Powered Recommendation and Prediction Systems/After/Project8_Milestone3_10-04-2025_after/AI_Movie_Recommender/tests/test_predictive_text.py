# test_predictive_text.py
# ===========================================================
# Unit tests for Milestone 2: Predictive Text Feature
# This test checks that the PredictiveText class returns up to
# five movie titles that match a given prefix, helping users
# find movies through intelligent, real-time suggestions.
# ===========================================================
import unittest
from src.milestone_2_predictive_text.predictive_text import PredictiveText

class TestPredictiveText(unittest.TestCase):
    """
    Unit tests for the PredictiveText class (Milestone 2).
    Verifies that title suggestions are returned based on prefix queries.
    """

    def setUp(self):
        # Initialize predictive text with the movie titles dataset
        self.predictor = PredictiveText("data/movies.csv")

    def test_suggest_returns_limited_matches(self):
        """
        Test that suggest() returns a list of up to 5 matching titles
        that start with the given input prefix.
        """
        suggestions = self.predictor.suggest("star")
        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 5)
        for title in suggestions:
            self.assertTrue(title.lower().startswith("star"))

if __name__ == "__main__":
    unittest.main()
