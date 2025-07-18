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

def validate_denominations(denominations):
    """
    Validate the coin denominations list.

    Args:
        denominations (list): List of coin denominations.

    Raises:
        ValueError: If denominations are not all positive integers.
    """
    if not all(isinstance(coin, int) and coin > 0 for coin in denominations):
        raise ValueError("All denominations must be positive integers.")