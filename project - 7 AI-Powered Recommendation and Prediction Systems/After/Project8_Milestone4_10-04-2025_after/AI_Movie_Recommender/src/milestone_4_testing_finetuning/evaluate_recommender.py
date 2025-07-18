# evaluate_recommender.py
# =============================================================
# Milestone 4: Evaluation and Fine-Tuning
# This module evaluates the performance of a movie recommender
# by comparing its predictions against real user ratings.
# It computes standard metrics: Precision, Recall, and F1-Score.
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

    # Determine the absolute path to the dataset
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_path = os.path.join(base_dir, "data", "ratings.csv")

    # Load all ratings and split them into training and test sets (80/20)
    ratings = pd.read_csv(data_path)
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    # Save training data to file (used by the recommender)
    train_path = os.path.join(base_dir, "data", "train.csv")
    train_data.to_csv(train_path, index=False)

    # Train the KNN-based recommender on the training set
    recommender = MovieRecommender(train_path)
    recommender.train()

    # Generate movie recommendations for the given user
    recommended_ids = recommender.recommend(user_id=user_id, n_recommendations=5)

    # Get the actual ratings of this user from the test set
    user_test_ratings = test_data[test_data["userId"] == user_id]

    if user_test_ratings.empty:
        print(f"No test data available for User {user_id}. Cannot evaluate.")
        return [] if return_predictions else None

    # Define "relevant" movies as those rated ≥ 4.0 in the test set
    relevant_movie_ids = user_test_ratings[user_test_ratings["rating"] >= 4.0]["movieId"].tolist()

    if not relevant_movie_ids:
        print(f"User {user_id} has no relevant ratings in the test set.")
        return [] if return_predictions else None

    # Determine how many recommended movies match the user's relevant movies
    hits = [movie_id for movie_id in recommended_ids if movie_id in relevant_movie_ids]

    # Compute evaluation metrics
    precision = len(hits) / len(recommended_ids) if recommended_ids else 0
    recall = len(hits) / len(relevant_movie_ids) if relevant_movie_ids else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Display evaluation results to the console
    print(f"Evaluating recommendations for User ID: {user_id}")
    print("Using an 80/20 train-test split on the full dataset.")
    print(f"{len(hits)} of {len(recommended_ids)} recommended movies matched those the user rated ≥ 4.0.")

    print_subsection("Evaluation Metrics")
    print(f"Precision: {precision:.2f} → Percentage of recommended movies that were relevant to the user.")
    print(f"Recall:    {recall:.2f} → Percentage of all relevant movies successfully recommended.")
    print(f"F1-Score:  {f1_score:.2f} → Balance between precision and recall.")

    # Return the predictions if requested (used by main.py or test)
    return recommended_ids if return_predictions else None

def print_subsection(title):
    """
    Prints a formatted yellow section header to distinguish output.
    """
    from colorama import Fore, Style
    print(f"\n{Fore.YELLOW}{'-'*10} {title} {'-'*10}{Style.RESET_ALL}")
