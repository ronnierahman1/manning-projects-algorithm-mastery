import unittest
import pandas as pd
import matplotlib.pyplot as plt
from src.milestone_4_system_evaluation.system_evaluation import evaluate_strategy, plot_trading_signals

class TestSystemEvaluation(unittest.TestCase):
    def setUp(self):
        """
        Setup a dummy DataFrame for testing system evaluation functions.
        This DataFrame has a DateTime index and a single 'Close' column.
        Also, setup a list of dummy trading decisions.
        """
        dates = pd.date_range(start="2023-01-01", periods=5, freq='D')
        self.data = pd.DataFrame({
            'Close': [100, 110, 105, 115, 120]
        }, index=dates)
        
        # Dummy decisions list for testing plot_trading_signals.
        # Each decision tuple is (action, date, price).
        self.decisions = [
            ('Buy', dates[1], 110),
            ('Sell', dates[2], 105),
            ('Buy', dates[3], 115)
        ]

    def test_evaluate_strategy(self):
        """
        Test the evaluate_strategy function.
        - It should plot cumulative returns with the title "Cumulative Returns from Historical Data".
        - Check that at least one x-axis gridline is visible.
        """
        # Override plt.show() to prevent the plot window from blocking the test.
        original_show = plt.show
        plt.show = lambda: None
        try:
            evaluate_strategy(self.data.copy(), 'Close')
            # Get current figure and axis.
            fig = plt.gcf()
            ax = fig.axes[0]
            # Verify the plot title.
            self.assertEqual(ax.get_title(), "Cumulative Returns from Historical Data")
            # Check for visible x-axis gridlines.
            x_gridlines = ax.get_xgridlines()
            grid_visible = any(line.get_visible() for line in x_gridlines)
            self.assertTrue(grid_visible, "Expected at least one visible x-axis gridline.")
        finally:
            plt.show = original_show
            plt.close('all')

    def test_plot_trading_signals(self):
        """
        Test the plot_trading_signals function.
        - It should create a plot with the title "Stock Price with Buy/Sell Signals".
        - The legend should include "Closing Price".
        """
        # Override plt.show() to prevent the plot window from blocking the test.
        original_show = plt.show
        plt.show = lambda: None
        try:
            plot_trading_signals(self.data.copy(), 'Close', self.decisions)
            fig = plt.gcf()
            ax = fig.axes[0]
            self.assertEqual(ax.get_title(), "Stock Price with Buy/Sell Signals")
            # Check that the legend contains "Closing Price".
            legend = ax.get_legend()
            legend_texts = [text.get_text() for text in legend.get_texts()] if legend is not None else []
            self.assertIn("Closing Price", legend_texts)
        finally:
            plt.show = original_show
            plt.close('all')

if __name__ == '__main__':
    unittest.main()
