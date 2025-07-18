"""
Utility functions for Greedy Coin Change Project.

This file contains helper functions used to load and manage coin denominations data.
"""

import pandas as pd

def load_denominations(file_path):
    """
    Load coin denominations from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing coin denominations.

    Returns:
        list: Sorted list of coin denominations in descending order.
    """
    # Read CSV file using pandas
    df = pd.read_csv(file_path)

    # Convert denominations to a sorted list (largest first)
    denominations = df['denomination'].tolist()
    denominations.sort(reverse=True)

    return denominations
