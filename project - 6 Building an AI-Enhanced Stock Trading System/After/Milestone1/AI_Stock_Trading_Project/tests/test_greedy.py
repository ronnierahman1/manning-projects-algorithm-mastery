import unittest
import pandas as pd
from src.milestone_2_greedy.greedy_algorithm import greedy_trade

class TestGreedyAlgorithm(unittest.TestCase):
    
    def test_greedy_trade_list(self):
        """
        Instructions for test_greedy_trade_list:
        
        1. Create a plain Python list of prices, for example: [100, 101, 99, 102].
        2. Call the greedy_trade function with this list.
        3. Assert that the output equals the expected list: ['Buy', 'Sell', 'Buy'].
        
        Replace the placeholder below with your test implementation.
        """
        # Write your test code here
        pass

    def test_greedy_trade_series(self):
        """
        Instructions for test_greedy_trade_series:
        
        1. Create a pandas Series with a DateTime index and sample prices, e.g., [100, 101, 99, 102].
           - Use pd.to_datetime to generate dates for the index.
        2. Call the greedy_trade function with the pandas Series.
        3. Define the expected output as a list of tuples, where each tuple contains:
             (action, date, price)
           For instance, if the Series is built with dates [d0, d1, d2, d3]:
             Expected output: [('Buy', d1, 101), ('Sell', d2, 99), ('Buy', d3, 102)]
        4. Use assertions to verify that the output matches the expected result.
        
        Replace the placeholder below with your test implementation.
        """
        # Write your test code here
        pass

if __name__ == '__main__':
    unittest.main()
