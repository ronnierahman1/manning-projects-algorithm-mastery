# predictive_text.py
# =============================================================
# Milestone 2: Predictive Text for Movie Titles
# TASK: Implement a suggestion system that returns movie titles
# starting with a user's input prefix (like an autocomplete).
# =============================================================

import pandas as pd

class PredictiveText:
    """
    A class that suggests movie titles based on a text prefix.
    """

    def __init__(self, movies_file):
        """
        Load movie titles from the provided CSV file.

        Parameters:
            movies_file (str): Path to the movies.csv file.
        """
        # Step 1: Load the CSV using pandas
        # Step 2: Extract the 'title' column as a list of strings
        # Step 3: Store it in self.movies
        # Write your code here
        pass

    def suggest(self, query):
        """
        Suggest up to 5 movie titles that start with the given prefix.

        Parameters:
            query (str): The beginning of a movie title

        Returns:
            list: Matching movie titles (up to 5)
        """
        # Step 4: Iterate over self.movies
        # Step 5: Select titles that start with the input prefix
        #         - Matching should be case-insensitive
        # Step 6: Return only the first 5 matches
        # Write your code here
        pass
