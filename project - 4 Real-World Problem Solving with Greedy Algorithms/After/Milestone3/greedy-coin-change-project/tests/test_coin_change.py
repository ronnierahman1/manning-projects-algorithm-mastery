"""
Unit Tests for Greedy Coin Change Calculator.

This test suite validates the correctness and robustness of the greedy_coin_change function.
It covers standard scenarios, edge cases, invalid inputs, and ensures the algorithm correctly handles exceptions.
"""

import unittest
from src.coin_change import greedy_coin_change
from src.utils import load_denominations, validate_denominations

class TestGreedyCoinChange(unittest.TestCase):
    """A set of tests for verifying the greedy coin change algorithm."""

    @classmethod
    def setUpClass(cls):
        """
        Load and validate coin denominations before any tests are run.
        This setup is executed only once for efficiency.
        """
        cls.denominations = load_denominations("data/coin_denominations.csv")
        validate_denominations(cls.denominations)

    def test_exact_change_standard(self):
        """Test a standard scenario with multiple denominations."""
        amount = 67
        expected = {25: 2, 10: 1, 5: 1, 1: 2}
        result = greedy_coin_change(amount, self.denominations)
        self.assertEqual(result, expected, f"Failed standard change test for {amount} cents.")

    def test_exact_change_large_amount(self):
        """Test algorithm correctness with a larger amount."""
        amount = 99
        expected = {25: 3, 10: 2, 1: 4}
        result = greedy_coin_change(amount, self.denominations)
        self.assertEqual(result, expected, f"Failed large amount test for {amount} cents.")

    def test_small_amount(self):
        """Test minimal amount ensuring smallest denomination is correctly used."""
        amount = 3
        expected = {1: 3}
        result = greedy_coin_change(amount, self.denominations)
        self.assertEqual(result, expected, f"Failed small amount test for {amount} cents.")

    def test_unreachable_exact_change(self):
        """Test scenario where exact change is impossible (should raise an error)."""
        denominations = [25, 10, 5]  # intentionally omitting '1' cent to trigger error
        validate_denominations(denominations)
        amount = 7
        with self.assertRaises(ValueError, msg=f"Failed to raise error when exact change isn't possible for {amount} cents."):
            greedy_coin_change(amount, denominations)

    def test_unsorted_denominations(self):
        """Ensure algorithm handles unsorted denominations properly."""
        denominations = [1, 25, 10, 5]  # unsorted intentionally
        validate_denominations(denominations)
        amount = 40
        expected = {25: 1, 10: 1, 5: 1}
        result = greedy_coin_change(amount, denominations)
        self.assertEqual(result, expected, f"Failed unsorted denominations test for {amount} cents.")

    def test_invalid_denominations(self):
        """Test validation logic for invalid denomination values (zero or negative)."""
        invalid_denominations = [25, 10, -5, 0]
        with self.assertRaises(ValueError, msg="Failed to detect invalid denominations containing zero or negative values."):
            validate_denominations(invalid_denominations)

if __name__ == "__main__":
    # Execute all tests
    unittest.main()
