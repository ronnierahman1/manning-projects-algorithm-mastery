import yfinance as yf
import matplotlib.pyplot as plt

# Milestone 4: Evaluating and Visualizing Trading Performance
# -------------------------------------------------------------
# This module provides two functions:
#
# 1. evaluate_strategy(data, close_col)
#    - This function calculates the percentage returns based on the specified close column
#      in the given DataFrame.
#    - It then computes the cumulative returns (by taking the cumulative product of (1 + returns))
#      and plots the resulting performance over time.
#
# 2. plot_trading_signals(data, close_col, decisions)
#    - This function plots the stock's closing price over time.
#    - It then overlays buy/sell signals based on the provided decisions list.
#      Each decision is a tuple containing (action, date, price).
#    - Buy signals are marked with a green upward-pointing marker and Sell signals with a red downward-pointing marker.
#
# Note: Both functions display plots using matplotlib's plt.show(), which will open a window
#       with the plot when called.
#
# Example usage for evaluate_strategy:
#   evaluate_strategy(data, 'Close')
#
# Example usage for plot_trading_signals:
#   plot_trading_signals(data, 'Close', decisions)

def evaluate_strategy(data, close_col):
    """
    This function should:
    1. Compute daily percentage returns from the close_col.
    2. Calculate the cumulative returns (by taking the cumulative product of 1 + returns).
    3. Plot the cumulative returns with the title "Cumulative Returns from Historical Data".
       - Set the x-axis label to "Date" and the y-axis label to "Cumulative Returns".
       - Enable gridlines on the plot.
    4. Display the plot using plt.show().
    """
    # Write your code here
    pass

def plot_trading_signals(data, close_col, decisions):
    """
    This function should:
    1. Create a new figure and axis using plt.subplots().
    2. Plot the closing price from the data (using data.index and data[close_col]) with the label "Closing Price".
    3. Iterate over each decision in the decisions list. For each decision tuple:
         - If the decision is 'Buy', plot an upward marker (e.g., a green upward triangle) at the corresponding date and price, and annotate with "Buy".
         - If the decision is 'Sell', plot a downward marker (e.g., a red downward triangle) at the corresponding date and price, and annotate with "Sell".
    4. Set the plot title to "Stock Price with Buy/Sell Signals".
         - Set the x-axis label to "Date" and the y-axis label to "Price".
         - Add a legend and enable gridlines.
    5. Display the plot using plt.show().
    """
    # Write your code here
    pass