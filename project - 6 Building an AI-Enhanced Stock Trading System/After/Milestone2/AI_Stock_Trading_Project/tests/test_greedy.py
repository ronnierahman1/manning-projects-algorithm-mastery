import unittest
import pandas as pd
from src.milestone_2_greedy.greedy_algorithm import greedy_trade

class TestGreedyAlgorithm(unittest.TestCase):
    def test_greedy_trade_list(self):
        """
        Test greedy_trade with a plain Python list.
        Input: [100, 101, 99, 102]
        Expected output: ['Buy', 'Sell', 'Buy']
        """
        prices = [100, 101, 99, 102]
        decisions = greedy_trade(prices)
        self.assertEqual(decisions, ['Buy', 'Sell', 'Buy'])

    def test_greedy_trade_series(self):
        """
        Test greedy_trade with a pandas Series.
        Input: Series with index as dates and values: [100, 101, 99, 102]
        Expected output: list of tuples:
            [('Buy', Timestamp('2023-01-01', ...), 101), ...]
        Note: The decision tuple format is (action, date, price).
        """
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
        prices = pd.Series([100, 101, 99, 102], index=dates)
        decisions = greedy_trade(prices)
        expected = [
            ('Buy', dates[1], 101),
            ('Sell', dates[2], 99),
            ('Buy', dates[3], 102)
        ]
        self.assertEqual(decisions, expected)

if __name__ == '__main__':
    unittest.main()
