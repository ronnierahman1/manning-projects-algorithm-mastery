import unittest
from src.milestone_1_quicksort.quicksort import quicksort

class TestQuickSort(unittest.TestCase):

    def test_quicksort_numbers(self):
        """
        Instructions for test_quicksort_numbers:

        1. Create a plain Python list of numbers, for example: [3, 6, 8, 10, 1, 2, 1].
        2. Call the quicksort function with the list.
        3. Assert that the returned list is sorted in ascending order.
           Expected output: [1, 1, 2, 3, 6, 8, 10].

        Replace the placeholder below with your test implementation.
        """
        # Write your test code here
        pass

    def test_quicksort_empty(self):
        """
        Instructions for test_quicksort_empty:

        1. Create an empty list.
        2. Call the quicksort function with the empty list.
        3. Assert that the returned list is also empty.

        Replace the placeholder below with your test implementation.
        """
        # Write your test code here
        pass

    def test_quicksort_stocks(self):
        """
        Instructions for test_quicksort_stocks:

        1. Create a list of dictionaries representing stock data.
           For example: [{'price': 5}, {'price': 3}, {'price': 10}].
        2. Call the quicksort function with the list and a key function that returns the 'price' field.
        3. Assert that the output list is correctly sorted by the 'price' value.
           Expected output: [{'price': 3}, {'price': 5}, {'price': 10}].

        Replace the placeholder below with your test implementation.
        """
        # Write your test code here
        pass

if __name__ == '__main__':
    unittest.main()
