class RecursiveHandling:
    """
    Implements recursive query handling for chatbot conversations.

    Features:
    - Handles nested user queries by breaking them down into subqueries.
    - Recursively processes each subquery to provide accurate responses.
    - Ensures the chatbot can handle multi-step, dependent questions.
    """

    def __init__(self, chatbot):
        """
        Initializes the recursive handling system with a chatbot instance.

        Parameters:
        -----------
        chatbot : AIChatbot
            The chatbot instance to process subqueries.
        """
        self.chatbot = chatbot

    def handle_recursive_query(self, query):
        """
        Processes a user query recursively, breaking it down into subqueries.

        Parameters:
        -----------
        query : str
            The user query, which may contain multiple nested questions.

        Returns:
        --------
        str
            The chatbot’s final response after handling all subqueries.
        """
        query = query.lower().strip()

        # If query starts with "nested", process it recursively
        if query.startswith("nested"):
            return self.handle_nested_query(query.replace("nested", "").strip())

        return self.chatbot.handle_query(query)

    def handle_nested_query(self, query):
        """
        Splits a nested query into multiple subqueries and processes them recursively.

        Parameters:
        -----------
        query : str
            The nested user query.

        Returns:
        --------
        str
            The chatbot’s combined response for all subqueries.
        """
        subqueries = self.split_into_subqueries(query)
        responses = [self.handle_recursive_query(subq) for subq in subqueries]
        return " | ".join(responses)

    def split_into_subqueries(self, query):
        """
        Splits a user query into multiple subqueries based on logical separators.

        Parameters:
        -----------
        query : str
            The original user query.

        Returns:
        --------
        list of str
            A list of individual subqueries extracted from the original query.
        """
        delimiters = [" and ", "; ", " & "]
        for delimiter in delimiters:
            if delimiter in query:
                return [subq.strip() for subq in query.split(delimiter) if subq.strip()]
        return [query]  # Return as a single-item list if no delimiters found


# Testing RecursiveHandling functionality
if __name__ == "__main__":
    from chatbot.chatbot import AIChatbot

    chatbot = AIChatbot("data/dev-v2.0.json")
    recursive_handler = RecursiveHandling(chatbot)

    # Sample nested queries
    test_queries = [
        "nested Explain recursion and dynamic programming",
        "nested What is AI and how does it work?",
        "nested Tell me about machine learning & what is deep learning?"
    ]

    print("Processing Nested Queries Recursively:")
    for query in test_queries:
        print(f"User: {query}")
        response = recursive_handler.handle_recursive_query(query)
        print(f"AIChatbot: {response}\n")
