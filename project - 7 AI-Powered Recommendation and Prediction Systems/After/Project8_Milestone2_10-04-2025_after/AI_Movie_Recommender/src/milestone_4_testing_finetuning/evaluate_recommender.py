# evaluate_recommender.py
# =============================================================
# Milestone 4: Evaluation and Fine-Tuning
# TASK: Implement the logic to evaluate the recommendation system.
# You will compute how well your model performs for a given user.
# =============================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.milestone_1_knn_recommendation.movie_recommender import MovieRecommender

def evaluate(user_id=1, return_predictions=False):
    """
    Evaluate the recommendation system by testing its predictions
    for a specific user and comparing those with the user's actual
    preferences from a test dataset.

    Parameters:
        user_id (int): The ID of the user to evaluate.
        return_predictions (bool): If True, return the list of recommended movie IDs.

    Returns:
        list or None: The list of recommended movie IDs if return_predictions is True,
                      otherwise None.
    """

    # Step 1: Load the dataset
    # ------------------------------------------------------------
    # a) Load ratings.csv from the 'data/' directory
    # b) Split the data into train (80%) and test (20%) using train_test_split
    # c) Save the train set as 'data/train.csv'
    # Write your code here

    # Step 2: Train your recommender model
    # ------------------------------------------------------------
    # a) Use your MovieRecommender from Milestone 1
    # b) Train the model on 'data/train.csv'
    # Write your code here

    # Step 3: Generate recommendations for the specified user
    # ------------------------------------------------------------
    # a) Call the recommend() method with the selected user ID
    # b) Store the output list of movie IDs
    # Write your code here

    # Step 4: Get user’s actual test ratings
    # ------------------------------------------------------------
    # a) Filter the test set to include only rows for the selected user
    # b) Define "relevant" as those movies with rating >= 4.0
    # c) Collect a list of those relevant movie IDs
    # Write your code here

    # Step 5: Evaluate recommendation quality
    # ------------------------------------------------------------
    # a) Count how many of the recommended movie IDs appear in the user's relevant movie list
    # b) Compute:
    #    - Precision = hits / number of recommendations
    #    - Recall = hits / number of relevant movies
    #    - F1-Score = harmonic mean of precision and recall
    # c) Print the results to the console
    # Write your code here

    # Step 6 (Optional): Return predictions if return_predictions is True
    # ------------------------------------------------------------
    # Return the list of movie IDs from your recommendation if requested
    # Write your code here
    pass  # Remove this after writing your code

def print_subsection(title):
    """
    Helper function to print formatted section headers in the console.
    """
    from colorama import Fore, Style
    print(f"\n{Fore.YELLOW}{'-'*10} {title} {'-'*10}{Style.RESET_ALL}")
