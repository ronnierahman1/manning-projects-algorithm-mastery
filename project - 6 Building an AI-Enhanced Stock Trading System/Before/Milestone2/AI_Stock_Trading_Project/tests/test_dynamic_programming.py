import unittest
import pandas as pd
from src.milestone_3_dynamic_programming.dynamic_programming import maximize_profit

class TestDynamicProgramming(unittest.TestCase):

    def test_empty_series(self):
        """
        Instructions:
        1. Create an empty pandas Series of floats.
        2. Call maximize_profit on the empty Series.
        3. Assert that the function returns 0.
        
        Replace the placeholder below with your test code.
        """
        # Write your test code here
        pass

    def test_single_element(self):
        """
        Instructions:
        1. Create a pandas Series with a single numeric element.
        2. Call maximize_profit on the Series.
        3. Assert that the function returns 0 (since no transaction can be made).
        
        Replace the placeholder below with your test code.
        """
        # Write your test code here
        pass

    def test_profit_positive(self):
        """
        Instructions:
        1. Create a pandas Series with values [7, 1, 5, 3, 6, 4].
        2. The expected maximum profit is 5 (buy at 1 and sell at 6).
        3. Assert that maximize_profit returns 5.
        
        Replace the placeholder below with your test code.
        """
        # Write your test code here
        pass

    def test_no_profit(self):
        """
        Instructions:
        1. Create a pandas Series with a monotonically decreasing sequence, e.g., [7, 6, 4, 3, 1].
        2. Assert that maximize_profit returns 0 (since no profit is possible).
        
        Replace the placeholder below with your test code.
        """
        # Write your test code here
        pass

    def test_profit_complex(self):
        """
        Instructions:
        1. Create a pandas Series with values [3, 2, 6, 1, 8, 3].
        2. The expected maximum profit is 7 (buy at 1 and sell at 8).
        3. Assert that maximize_profit returns 7.
        
        Replace the placeholder below with your test code.
        """
        # Write your test code here
        pass

if __name__ == '__main__':
    unittest.main()
