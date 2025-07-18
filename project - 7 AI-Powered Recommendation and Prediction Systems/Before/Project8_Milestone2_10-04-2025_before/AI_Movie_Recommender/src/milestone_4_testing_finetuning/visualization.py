# visualization.py
# =============================================================
# Milestone 4: Evaluation and Visualization
# This module is responsible for visualizing performance metrics
# of the recommendation system using a bar chart.
# =============================================================

import matplotlib.pyplot as plt

def visualize_performance(metrics):
    """
    Display a bar chart of evaluation metrics for the recommender system.

    Parameters:
        metrics (dict): A dictionary of metric names and their values.
                        Example: {'Precision': 0.8, 'Recall': 0.7, 'F1-Score': 0.75}
    """
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Recommender System Performance')
    plt.ylabel('Scores')
    plt.xlabel('Metrics')
    plt.tight_layout()
    plt.show()

# If this file is run directly, display a sample chart
if __name__ == "__main__":
    sample_metrics = {'Precision': 0.8, 'Recall': 0.7, 'F1-Score': 0.75}
    visualize_performance(sample_metrics)
