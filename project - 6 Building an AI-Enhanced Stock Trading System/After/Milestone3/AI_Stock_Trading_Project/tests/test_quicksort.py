import unittest
from src.milestone_1_quicksort.quicksort import quicksort

class TestQuickSort(unittest.TestCase):
    def test_quicksort_numbers(self):
        """
        Test quicksort with a list of numbers.
        Input: [3,6,8,10,1,2,1]
        Expected output: [1,1,2,3,6,8,10]
        """
        self.assertEqual(quicksort([3,6,8,10,1,2,1]), [1,1,2,3,6,8,10])

    def test_quicksort_empty(self):
        """
        Test quicksort with an empty list.
        Expected output: []
        """
        self.assertEqual(quicksort([]), [])

    def test_quicksort_stocks(self):
        """
        Test quicksort with a list of dictionaries representing stocks.
        Sorting is based on the 'price' key.
        Input: [{'price':5}, {'price':3}, {'price':10}]
        Expected output: [{'price':3}, {'price':5}, {'price':10}]
        """
        stocks = [{'price':5}, {'price':3}, {'price':10}]
        sorted_stocks = quicksort(stocks, key=lambda x: x['price'])
        self.assertEqual(sorted_stocks, [{'price':3}, {'price':5}, {'price':10}])

if __name__ == '__main__':
    unittest.main()
