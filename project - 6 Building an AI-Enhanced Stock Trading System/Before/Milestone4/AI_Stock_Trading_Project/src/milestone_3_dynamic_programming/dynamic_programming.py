import numpy as np

# Milestone 3: Dynamic Programming for Profit Maximization
# ---------------------------------------------------------
# This module implements a function that calculates the maximum profit achievable
# from a single buy-and-sell transaction, given a sequence of stock prices.
#
# The algorithm works as follows:
#   - If the input prices series is empty, return 0 (no profit).
#   - Create an array "dp" to track the maximum profit achievable up to each index.
#   - Keep track of the minimum price seen so far.
#   - For each price in the series, update the minimum price and compute the potential profit
#     (current price minus minimum price), and update the dp array with the maximum profit seen so far.
#
# Example usage:
#   import pandas as pd
#   prices = pd.Series([7, 1, 5, 3, 6, 4])
#   profit = maximize_profit(prices)
#   print(profit)  # Output: 5 (buy at 1, sell at 6)

def maximize_profit(prices):
    if prices.empty:
        return 0
    n = len(prices)
    dp = np.zeros(n)
    # Initialize the minimum price with the first price in the series
    min_price = prices.iloc[0]
    for i in range(1, n):
        # Update the minimum price so far
        min_price = min(min_price, prices.iloc[i])
        # Calculate the maximum profit achievable up to index i
        dp[i] = max(dp[i - 1], prices.iloc[i] - min_price)
    # Return the last element in dp array, which holds the maximum profit overall
    return int(dp[-1])

