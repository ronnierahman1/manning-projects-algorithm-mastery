import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import functions from the src folder:
from src.milestone_1_quicksort.quicksort import quicksort
from src.milestone_2_greedy.greedy_algorithm import greedy_trade
from src.milestone_3_dynamic_programming.dynamic_programming import maximize_profit
from src.milestone_4_system_evaluation.system_evaluation import evaluate_strategy, plot_trading_signals

# MAIN FUNCTION
def main():
    print("Starting AI-Enhanced Stock Trading System...", flush=True)
    
    # Load historical data from CSV.
    # The CSV file contains extra header rows which we skip by using skiprows=3.
    # We then assign explicit column names and parse the "Date" column as a datetime type,
    # setting it as the DataFrame index.
    print("Loading historical data from CSV...", flush=True)
    try:
        data = pd.read_csv(
            'data/historical_stock_data.csv',     # CSV file with AAPL data
            skiprows=3,                           # Skip extra header rows
            header=None,                          # No header row in the remaining data
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
            index_col="Date"                      # Set the Date column as index
        )
    except Exception as e:
        print(f"Error loading historical data CSV: {e}", flush=True)
        sys.exit(1)
    
    # Check if the data is loaded correctly.
    if data.empty:
        print("Historical data CSV is empty. Please check the file.", flush=True)
        sys.exit(1)
    else:
        print(f"Loaded {len(data)} rows of historical data from CSV.", flush=True)
    
    # Determine which column to use for the closing price data.
    # Prefer 'Close'; if not found, try 'Adj Close'.
    if 'Close' in data.columns:
        close_col = 'Close'
    elif 'Adj Close' in data.columns:
        close_col = 'Adj Close'
    else:
        print("No valid close column found in CSV.", flush=True)
        sys.exit(1)
    
    # Drop rows with missing closing prices to avoid errors in further processing.
    data = data.dropna(subset=[close_col])
    
    # Milestone 1: Sort data by closing price using quicksort.
    print("\nMilestone 1: Sorting data by closing price...", flush=True)
    # Convert the DataFrame to a list of dictionaries.
    data_records = data.reset_index().to_dict(orient='records')
    # Use the quicksort function with the key set to the 'Close' price.
    sorted_data = quicksort(data_records, key=lambda x: x.get('Close'))
    print("Top 5 days with lowest closing prices after sorting:", flush=True)
    for record in sorted_data[:5]:
        print(record, flush=True)
    
    # Milestone 2: Generate greedy trade decisions based on historical closing prices.
    print("\nMilestone 2: Generating greedy trade decisions...", flush=True)
    close_prices = data[close_col]
    # Call the greedy_trade function; it returns a list of decisions.
    decisions = greedy_trade(close_prices)
    print("First 5 trade decisions (Greedy Algorithm):", flush=True)
    for decision in decisions[:5]:
        print(decision, flush=True)
    
    # Milestone 3: Calculate maximum profit using Dynamic Programming.
    print("\nMilestone 3: Calculating maximum profit...", flush=True)
    profit = maximize_profit(close_prices)
    print(f"Maximum profit achievable (buy-low sell-high once): ${profit}", flush=True)
    
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

# Run the main function when executed as a script.
if __name__ == '__main__':
    main()
