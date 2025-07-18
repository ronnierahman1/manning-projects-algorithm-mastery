class GreedyPriority:
    """
    Implements a greedy algorithm to prioritize chatbot responses based on urgency.

    Features:
    - Assigns priority levels to queries based on keywords.
    - Ensures the chatbot processes high-priority queries first.
    - Optimizes response time by reducing delays for critical queries.
    """

    def __init__(self):
        """
        Initializes the priority system with predefined levels.
        """
        self.priority_map = {
            "urgent": 1,       # Highest priority
            "important": 2,    # Medium priority
            "general": 3       # Lowest priority (default)
        }

    def get_priority(self, query):
        """
        Determines the priority of a given query.

        Parameters:
        -----------
        query : str
            User query to classify.

        Returns:
        --------
        int
            The assigned priority (1 = highest, 3 = lowest).
        """
        normalized_query = query.lower().strip()
        if "urgent" in normalized_query:
            return self.priority_map["urgent"]
        elif "important" in normalized_query:
            return self.priority_map["important"]
        else:
            return self.priority_map["general"]

    def sort_queries_by_priority(self, queries):
        """
        Sorts a list of queries based on priority using a greedy approach.

        Parameters:
        -----------
        queries : list of str
            User queries to be prioritized.

        Returns:
        --------
        list of str
            Queries sorted by priority (highest first).
        """
        return sorted(queries, key=self.get_priority)

# Testing GreedyPriority functionality
if __name__ == "__main__":
    priority_manager = GreedyPriority()

    # Sample queries with mixed priority levels
    test_queries = [
        "urgent: I need help immediately",
        "important: Tell me about machine learning",
        "general: What is AI?",
        "urgent: Explain deep learning",
        "general: How does a chatbot work?",
        "important: What is reinforcement learning?"
    ]

    # Sort queries by priority
    sorted_queries = priority_manager.sort_queries_by_priority(test_queries)

    print("Queries sorted by priority (Highest first):")
    for query in sorted_queries:
        print(f"- {query}")
