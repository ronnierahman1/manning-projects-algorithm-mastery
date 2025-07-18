import unittest
import pandas as pd
from src.milestone_3_dynamic_programming.dynamic_programming import maximize_profit

class TestDynamicProgramming(unittest.TestCase):
    def test_empty_series(self):
        """
        Test that an empty pandas Series returns 0 profit.
        Expected: maximize_profit(pd.Series([], dtype=float)) -> 0
        """
        prices = pd.Series([], dtype=float)
        self.assertEqual(maximize_profit(prices), 0)
    
    def test_single_element(self):
        """
        Test that a Series with a single element returns 0 profit 
        since there's no opportunity to sell after buying.
        Expected: maximize_profit(pd.Series([100])) -> 0
        """
        prices = pd.Series([100])
        self.assertEqual(maximize_profit(prices), 0)
    
    def test_profit_positive(self):
        """
        Test with the classic example where the maximum profit is achieved 
        by buying at the minimum price and selling at a later higher price.
        For the input [7, 1, 5, 3, 6, 4], maximum profit should be 5.
        Expected: maximize_profit(pd.Series([7, 1, 5, 3, 6, 4])) -> 5
        """
        prices = pd.Series([7, 1, 5, 3, 6, 4])
        self.assertEqual(maximize_profit(prices), 5)
    
    def test_no_profit(self):
        """
        Test a monotonically decreasing sequence, where no profit is possible.
        Expected: maximize_profit(pd.Series([7, 6, 4, 3, 1])) -> 0
        """
        prices = pd.Series([7, 6, 4, 3, 1])
        self.assertEqual(maximize_profit(prices), 0)
    
    def test_profit_complex(self):
        """
        Test a more complex scenario: for prices [3, 2, 6, 1, 8, 3], 
        the best profit is 7 (buy at 1 and sell at 8).
        Expected: maximize_profit(pd.Series([3, 2, 6, 1, 8, 3])) -> 7
        """
        prices = pd.Series([3, 2, 6, 1, 8, 3])
        self.assertEqual(maximize_profit(prices), 7)

if __name__ == '__main__':
    unittest.main()
