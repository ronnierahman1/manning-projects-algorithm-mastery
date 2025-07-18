import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import functions from the src folder for various milestones:
from src.milestone_1_quicksort.quicksort import quicksort
from src.milestone_2_greedy.greedy_algorithm import greedy_trade
from src.milestone_3_dynamic_programming.dynamic_programming import maximize_profit
from src.milestone_4_system_evaluation.system_evaluation import evaluate_strategy, plot_trading_signals

# MAIN FUNCTION
def main():
    print("Starting AI-Enhanced Stock Trading System...", flush=True)
    
    # Step 1: Load Historical Data from CSV.
    # -----------------------------------------
    # - The CSV file contains extra header rows. We skip these using skiprows=3.
    # - We then assign explicit column names and parse the "Date" column as a datetime type.
    # - Finally, we set the "Date" as the DataFrame index.
    print("Loading historical data from CSV...", flush=True)
    try:
        data = pd.read_csv(
            'data/historical_stock_data.csv',
            skiprows=3,
            header=None,
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"
        )
    except Exception as e:
        print(f"Error loading historical data CSV: {e}", flush=True)
        sys.exit(1)
    
    if data.empty:
        print("Historical data CSV is empty. Please check the file.", flush=True)
        sys.exit(1)
    else:
        print(f"Loaded {len(data)} rows of historical data from CSV.", flush=True)
    
    # Determine the closing price column to use:
    if 'Close' in data.columns:
        close_col = 'Close'
    elif 'Adj Close' in data.columns:
        close_col = 'Adj Close'
    else:
        print("No valid close column found in CSV.", flush=True)
        sys.exit(1)
    
    # Remove rows with missing closing prices.
    data = data.dropna(subset=[close_col])
    
    # Milestone 1: Data Sorting with Quicksort
    # -----------------------------------------
    # Instructions:
    # 1. Convert the DataFrame 'data' into a list of dictionaries.
    #    (Hint: Use data.reset_index().to_dict(orient='records') to achieve this.)
    # 2. Call the 'quicksort' function with the list and a key function that extracts the 'Close' value for sorting.
    # 3. Print the top 5 records from the sorted list to verify that the data is ordered correctly.
    #
    # Write your code here:
    pass

    # Milestone 2: Real-Time Trading Decisions with a Greedy Algorithm
    # ---------------------------------------------------------------
    # Instructions:
    # 1. Extract the closing prices from 'data' using the variable 'close_col'.
    # 2. Pass these prices to the 'greedy_trade' function, which should generate a list of trade decisions.
    # 3. Print the first 5 decisions returned by the function to check that the decisions are generated correctly.
    #
    # Write your code here:
    pass
    
    # Milestone 3: Optimizing Trading Strategies with Dynamic Programming
    # --------------------------------------------------------------------
    print("\nMilestone 3: Calculating maximum profit...", flush=True)
    # Instructions:
    # 1. Using the 'close_prices' pandas Series (loaded from the CSV earlier), 
    #    call the maximize_profit function to compute the maximum profit.
    # 2. The maximize_profit function should determine the best profit possible from a single buy and sell transaction.
    # 3. Print the result in the format:
    #       "Maximum profit achievable (buy-low sell-high once): $<profit>"
    #
    # Replace the placeholder below with your code that calls maximize_profit and prints the profit.
    # Write your code here:
    pass
    
    # Milestone 4: Evaluating and Visualizing Trading Performance
    # -------------------------------------------------------------
    # In this milestone, you will complete two major tasks:
    #
    # Task 1: Evaluate Strategy
    #   - Calculate the daily percentage returns from the 'close_col' data.
    #   - Compute the cumulative returns by taking the cumulative product of (1 + daily returns).
    #   - Plot the cumulative returns:
    #         * Title: "Cumulative Returns from Historical Data"
    #         * X-axis label: "Date"
    #         * Y-axis label: "Cumulative Returns"
    #         * Enable gridlines.
    #   - Display the plot using plt.show().
    #
    # Task 2: Plot Trading Signals
    #   - Plot the stock's closing price over time.
    #   - Overlay the trading signals from the 'decisions' list on the price chart.
    #         * Each decision is a tuple: (action, date, price).
    #         * Mark "Buy" signals with a green upward marker (e.g., '^') and annotate with "Buy".
    #         * Mark "Sell" signals with a red downward marker (e.g., 'v') and annotate with "Sell".
    #   - Set the plot title to "Stock Price with Buy/Sell Signals".
    #   - Add axis labels, legend, and gridlines.
    #   - Display the plot using plt.show().
    #
    # Replace the code below with your implementation for Milestone 4.
    # Write your code here:
    pass

    print("Program completed successfully.", flush=True)

if __name__ == '__main__':
    main()
