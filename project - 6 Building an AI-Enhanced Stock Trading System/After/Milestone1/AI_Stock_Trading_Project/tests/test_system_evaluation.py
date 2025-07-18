import unittest
import pandas as pd
import matplotlib.pyplot as plt
from src.milestone_4_system_evaluation.system_evaluation import evaluate_strategy, plot_trading_signals

class TestSystemEvaluation(unittest.TestCase):

    def test_evaluate_strategy(self):
        """
        Instructions for test_evaluate_strategy:
        
        1. Create a dummy DataFrame (named 'data') with a DateTime index.
           - The DataFrame should contain at least one column named 'Close' with numerical values.
           - For example, use pd.date_range to generate dates and a list of close prices.
        
        2. Call evaluate_strategy() using a copy of the dummy DataFrame and the column name 'Close'.
           - This function should calculate the cumulative returns and generate a plot.
        
        3. Retrieve the current matplotlib figure (using plt.gcf()) and the first axis (using fig.axes[0]).
        
        4. Write an assertion to check that the axis title is exactly "Cumulative Returns from Historical Data".
           - (Hint: Use ax.get_title())
        
        5. Write an assertion to check that at least one x-axis gridline is visible.
           - (Hint: Retrieve grid lines with ax.get_xgridlines() and check if any line's get_visible() returns True)
        
        6. Finally, close the figure (using plt.close('all')).

        Replace the placeholder below with your implementation.
        """
        # Write your test code here
        pass

    def test_plot_trading_signals(self):
        """
        Instructions for test_plot_trading_signals:
        
        1. Create a dummy DataFrame (named 'data') with a DateTime index and a column 'Close'.
           - Use pd.date_range to generate dates and a list of close prices.
        
        2. Create a dummy decisions list.
           - The decisions list should be a list of tuples in the form: (action, date, price)
           - For example: [('Buy', date, price), ('Sell', date, price), ...]
        
        3. Call plot_trading_signals() with a copy of the dummy DataFrame, the string 'Close', and the decisions list.
           - This function should generate a plot overlaying the buy/sell signals on the stock price plot.
        
        4. Retrieve the current matplotlib figure (using plt.gcf()) and the first axis (using fig.axes[0]).
        
        5. Write an assertion to verify that the axis title is exactly "Stock Price with Buy/Sell Signals".
           - (Hint: Use ax.get_title())
        
        6. Write an assertion to check that the plot legend includes the label "Closing Price".
           - (Hint: Retrieve the legend from the axis and check the text of the legend entries)
        
        7. Close the figure using plt.close('all').

        Replace the placeholder below with your implementation.
        """
        # Write your test code here
        pass

if __name__ == '__main__':
    unittest.main()
