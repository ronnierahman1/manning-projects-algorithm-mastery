"""
dynamic_context.py

Uses dynamic programming to store and reuse responses for context-aware queries.
"""

class DynamicContextCache:
    """
    Implements a cache to store and retrieve chatbot responses, ensuring
    dynamic programming principles are applied to optimize repeated queries.

    Attributes:
    -----------
    cache : dict
        Stores query-response pairs for quick retrieval.
    """
    def __init__(self):
        self.cache = {}

    def get(self, query):
        """Retrieve a cached response for a query if it exists."""
        return self.cache.get(query)

    def set(self, query, response):
        """Cache the response for a given query."""
        self.cache[query] = response
