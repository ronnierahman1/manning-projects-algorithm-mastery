"""
Milestone 3: Implement a Greedy Algorithm for Optimizing Response Time

This module provides a simple greedy strategy to prioritize user queries.
The chatbot can choose to respond to "urgent" questions earlier or more quickly,
by assigning priorities to queries based on detected keywords or sentiment.
"""

from typing import List, Tuple


class GreedyPriority:
    """
    A greedy algorithm to determine query priority based on urgency, type, or keywords.
    Lower number means higher priority (i.e., priority 1 is most urgent).
    """

    def __init__(self):
        """
        Initialize common priority rules and weights.
        You can define keyword groups like 'urgent', 'greeting', etc.
        """
        # TODO: Define your keyword buckets here.
        # Example:
        # self.urgent_keywords = ["emergency", "urgent", "immediately"]
        # self.casual_keywords = ["hello", "hi", "bye", "thanks"]
        pass

    def get_priority(self, query: str) -> int:
        """
        Determine the priority of a single query.

        Args:
            query (str): The user query to analyze

        Returns:
            int: An integer priority score (1 = highest priority, 5 = lowest)
        """
        # TODO: Analyze the query string using the defined keyword groups
        # and return an integer representing the priority level.

        # EXAMPLE LOGIC TO FOLLOW:
        # if contains urgent keyword:
        #     return 1
        # elif contains greeting:
        #     return 2
        # else:
        #     return 3
        return 3  # default fallback (normal priority)

    def sort_queries_by_priority(self, queries: List[str]) -> List[Tuple[int, str]]:
        """
        Sort a list of queries based on their priority using greedy selection.

        Args:
            queries (List[str]): List of user query strings

        Returns:
            List[Tuple[int, str]]: List of tuples with (priority, query), sorted by priority ascending
        """
        # TODO: Use self.get_priority on each query and sort them
        # Example result: [(1, 'emergency call'), (2, 'hello'), (3, 'how are you?')]

        return []  # Replace with sorted output

    def record_query_stats(self, query: str):
        """
        (Optional) Record stats about incoming queries to simulate optimization tracking.
        This is useful for later analytics but is not required to complete this milestone.

        Args:
            query (str): The query to record
        """
        # TODO: Implement simple stat tracking (e.g., keyword frequency or type counts)
        pass

    def get_optimization_insights(self) -> str:
        """
        (Optional) Return a summary of greedy optimization decisions made so far.
        This helps evaluate the effectiveness of the prioritization logic.

        Returns:
            str: Human-readable explanation of current stats
        """
        # TODO: Summarize stats from record_query_stats()
        return "Greedy optimization insights are not yet implemented."
