"""
Utility functions for Greedy Coin Change Project.

You'll implement the helper functions used to load and manage coin denominations data.
"""

import pandas as pd

def load_denominations(file_path):
    """
    STEP 1:
    Load coin denominations from a CSV file.

    Instructions:
    - Read the CSV file provided at 'file_path' using pandas.
    - Extract the column 'denomination' into a Python list.
    - Sort this list in descending order (largest coin first).
    - Return the sorted list of denominations.

    Example:
    Given denominations [5, 25, 1, 10], your function should return [25, 10, 5, 1].
    """

    # Write your code here
