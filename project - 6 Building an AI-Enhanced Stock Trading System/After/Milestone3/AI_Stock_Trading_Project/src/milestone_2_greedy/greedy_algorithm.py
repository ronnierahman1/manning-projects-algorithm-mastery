# Milestone 2: Greedy Algorithm for Real-Time Trading Decisions
# ---------------------------------------------------------------
# This module defines the function "greedy_trade" to generate simple trading decisions.
# The function compares consecutive prices and, if the price increases from one step to the next,
# it signals a "Buy" decision; otherwise, it signals "Sell."
#
# The function supports two types of inputs:
#   1. A plain Python list: In this case, it returns a list of decision strings.
#   2. A pandas Series: In this case, it returns a list of tuples, each containing:
#         (decision, corresponding date from the Series index, price)
#
# Example usage with a list:
#   decisions = greedy_trade([100, 101, 99, 102])
#   print(decisions)  # Output: ['Buy', 'Sell', 'Buy']
#
# Example usage with a pandas Series:
#   import pandas as pd
#   dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
#   prices = pd.Series([100, 101, 99, 102], index=dates)
#   decisions = greedy_trade(prices)
#   # Output:
#   # [('Buy', Timestamp('2023-01-02 00:00:00'), 101),
#   #  ('Sell', Timestamp('2023-01-03 00:00:00'), 99),
#   #  ('Buy', Timestamp('2023-01-04 00:00:00'), 102)]

def greedy_trade(prices):
    decisions = []
    # Check if 'prices' is a simple list
    if isinstance(prices, list):
        # For plain lists, generate only the decision strings
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                decisions.append("Buy")
            else:
                decisions.append("Sell")
    else:
        # Assume prices is a pandas Series
        # Use .iloc for position-based access and prices.index for date values
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i - 1]:
                decisions.append(("Buy", prices.index[i], prices.iloc[i]))
            else:
                decisions.append(("Sell", prices.index[i], prices.iloc[i]))
    return decisions
