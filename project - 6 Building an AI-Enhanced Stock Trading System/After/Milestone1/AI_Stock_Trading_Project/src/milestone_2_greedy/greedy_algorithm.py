def greedy_trade(prices):
    """
    Instructions for implementing the greedy_trade function:
    
    This function generates trading decisions based on consecutive price comparisons.
    
    If 'prices' is a plain Python list:
      1. Loop through the list starting from index 1.
      2. Compare each price with its previous price.
      3. Append "Buy" to the decisions list if the current price is higher than the previous price;
         otherwise, append "Sell".
    
    If 'prices' is a pandas Series:
      1. Loop through the series using position-based indexing (use .iloc).
      2. Compare each price (prices.iloc[i]) with the previous price (prices.iloc[i-1]).
      3. For each comparison, append a tuple to the decisions list in the format:
            (action, corresponding_date, current_price)
         where:
            - action is "Buy" if prices.iloc[i] > prices.iloc[i-1], or "Sell" otherwise;
            - corresponding_date is taken from prices.index[i];
            - current_price is prices.iloc[i].
    
    The function should return a list of decisions.
    
    Replace the placeholder below with your implementation.
    """
    # Write your code here
    pass
