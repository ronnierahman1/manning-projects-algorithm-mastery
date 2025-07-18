"""
Milestone 1: Recursive Query Handling (BEFORE VERSION)

This module is intended to be completed by learners as part of Milestone 1.
The structure and method headers are provided. Learners will implement the recursive
query handling logic based on the provided instructions.
"""

import re
from typing import List


class RecursiveHandling:
    def __init__(self, chatbot):
        """
        Initialize the RecursiveHandling with a reference to the chatbot
        and define the recursion limit and patterns for nested queries.
        """
        self.chatbot = chatbot
        self.max_recursion_depth = 5
        self.recursion_patterns = [
            r'tell me about (.*) and (.*)',
            r'what about (.*) and (.*)',
            r'explain (.*) and (.*)',
            r'describe (.*) and (.*)',
            r'(.*) and also (.*)',
            r'(.*) as well as (.*)'
        ]

    def handle_recursive_query(self, query: str, depth: int = 0) -> str:
        """
        Entry point to detect and delegate recursive/nested queries.

        Args:
            query (str): Input from user
            depth (int): Internal tracker for recursion depth

        Returns:
            str: Response to the query
        """
        # TODO: Step 1 – Guard against too deep recursion
        # Example: if depth >= self.max_recursion_depth:
        #              return "Query too complex..."

        # TODO: Step 2 – Use _is_nested_query() to check if query is recursive

        # TODO: Step 3 – If recursive, delegate to handle_nested_query

        # TODO: Step 4 – If not, delegate to chatbot.knowledge_base.get_answer()

        pass  # Replace this after implementing

    def handle_nested_query(self, query: str, depth: int = 0) -> str:
        """
        Decomposes a compound query and recursively handles each part.

        Args:
            query (str): Complex user question
            depth (int): Recursion level

        Returns:
            str: Structured multi-part answer
        """
        # TODO: Step 1 – Check max depth, return fallback message if exceeded

        # TODO: Step 2 – Use split_into_subqueries() to break query

        # TODO: Step 3 – If more than one subquery, process each with recursion

        # TODO: Step 4 – Format each answer with numbering and combine

        # TODO: Step 5 – Return joined result with optional intro message

        pass  # Replace this after implementing

    def split_into_subqueries(self, query: str) -> List[str]:
        """
        Splits complex query using regex, conjunctions, and heuristics.

        Args:
            query (str): Raw user input

        Returns:
            List[str]: A list of sub-queries (may contain only one item)
        """
        # TODO: Step 1 – Try all patterns in self.recursion_patterns using re.search()

        # TODO: Step 2 – Try to split using conjunctions like 'and', 'or', 'plus'

        # TODO: Step 3 – Split on punctuation marks and detect question patterns

        # TODO: Step 4 – Filter parts to ensure meaningful subqueries

        # Return the full query if no split is viable
        return [query]

    def _is_nested_query(self, query: str) -> bool:
        """
        Checks for keywords and patterns suggesting compound queries.

        Args:
            query (str): User input

        Returns:
            bool: True if query is recursive/compound
        """
        nested_indicators = [
            ' and ', ' & ', '; ', ' or ', ' plus ',
            'also', 'additionally', 'furthermore', 'moreover',
            'what about', 'how about', 'tell me about',
            'explain both', 'describe both', 'compare'
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in nested_indicators)
