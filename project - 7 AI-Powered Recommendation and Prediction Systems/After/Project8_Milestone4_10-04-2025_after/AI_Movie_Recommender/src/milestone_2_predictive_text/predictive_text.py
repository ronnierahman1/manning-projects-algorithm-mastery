# predictive_text.py
# =============================================================
# Milestone 2: Predictive Text for Movie Titles
# This module allows the user to type part of a title and get
# smart suggestions based on prefix matching.
# =============================================================

import pandas as pd

class PredictiveText:
    """
    Implements a simple predictive text feature that suggests
    movie titles based on a given text prefix.
    """

    def __init__(self, movies_file):
        """
        Initialize the class by loading movie titles.

        Parameters:
            movies_file (str): Path to the CSV file with a 'title' column.
        """
        self.movies = pd.read_csv(movies_file)['title'].tolist()

    def suggest(self, query):
        """
        Suggest up to 5 movie titles that start with the input query.

        Parameters:
            query (str): The partial movie title input by the user

        Returns:
            List of matching movie titles (max 5)
        """
        return [title for title in self.movies if title.lower().startswith(query.lower())][:5]
