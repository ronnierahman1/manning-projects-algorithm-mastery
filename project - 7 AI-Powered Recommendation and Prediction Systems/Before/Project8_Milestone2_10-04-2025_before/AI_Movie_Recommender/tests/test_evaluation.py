# test_evaluation.py
# ===========================================================
# Unit tests for Milestone 4: Evaluation and Visualization
# This test ensures that the evaluate() function correctly returns
# a list of predicted movie IDs for a given user, which are used
# to assess the recommendation system’s accuracy.
# ===========================================================

import unittest
from src.milestone_4_testing_finetuning.evaluate_recommender import evaluate

class TestEvaluation(unittest.TestCase):
    def test_evaluation_returns_recommendations(self):
        """
        Test that the evaluate function returns a list of 5 movie IDs
        when return_predictions=True. These should be numeric and match
        the expected type and structure of recommendations.
        """
        recs = evaluate(user_id=1, return_predictions=True)
        self.assertIsInstance(recs, list)
        self.assertEqual(len(recs), 5)
        self.assertTrue(all(isinstance(mid, (int, float)) for mid in recs))

if __name__ == "__main__":
    unittest.main()
