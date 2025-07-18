class DynamicContext:
    """
    Implements dynamic programming to store and retrieve responses based on previous conversations,
    improving the chatbot's ability to maintain context.

    Features:
    - Caches previously asked questions and responses for quick retrieval.
    - Enhances chatbot memory, allowing it to recall past interactions.
    - Prevents redundant processing of repeated queries.
    """

    def __init__(self):
        """
        Initializes the dynamic context cache.
        """
        self.context_cache = {}  # Stores past queries and their responses
        self.conversation_history = []  # Maintains ordered history of conversations

    def store_in_cache(self, query, response):
        """
        Stores the response for a given query in the cache.

        Parameters:
        -----------
        query : str
            User query.
        response : str
            Response generated for the query.
        """
        normalized_query = query.lower().strip().rstrip("?")  # Normalize query format
        self.context_cache[normalized_query] = response
        self.conversation_history.append((query, response))

    def retrieve_from_cache(self, query):
        """
        Retrieves a cached response for a previously asked query.

        Parameters:
        -----------
        query : str
            User query to search in the cache.

        Returns:
        --------
        str or None
            Cached response if available, otherwise None.
        """
        normalized_query = query.lower().strip().rstrip("?")  # Normalize query
        return self.context_cache.get(normalized_query, None)

    def get_last_response(self):
        """
        Retrieves the last response from the conversation history.

        Returns:
        --------
        str or None
            Last chatbot response if available, otherwise None.
        """
        return self.conversation_history[-1][1] if self.conversation_history else None

    def has_context(self, query):
        """
        Checks if the chatbot has context for a given query.

        Parameters:
        -----------
        query : str
            User query.

        Returns:
        --------
        bool
            True if context exists, False otherwise.
        """
        normalized_query = query.lower().strip().rstrip("?")
        return normalized_query in self.context_cache

# Testing DynamicContext functionality
if __name__ == "__main__":
    context_manager = DynamicContext()

    # Simulate storing queries and responses
    context_manager.store_in_cache("What is AI?", "AI stands for Artificial Intelligence.")
    context_manager.store_in_cache("Explain deep learning", "Deep learning is a subset of machine learning that uses neural networks with multiple layers.")

    # Retrieving cached responses
    test_queries = [
        "What is AI?",
        "Explain deep learning",
        "How does AI work?"
    ]

    for query in test_queries:
        cached_response = context_manager.retrieve_from_cache(query)
        if cached_response:
            print(f"User: {query}")
            print(f"AIChatbot (cached): {cached_response}")
        else:
            print(f"User: {query}")
            print("AIChatbot: No cached response found. Processing...")
        print("-" * 50)
