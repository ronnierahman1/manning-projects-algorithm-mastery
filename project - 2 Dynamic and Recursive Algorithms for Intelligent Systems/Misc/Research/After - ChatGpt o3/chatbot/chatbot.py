import random
from datetime import datetime
from chatbot.knowledge_base import KnowledgeBase
from ai.ai_module import AIModule


class AIChatbot:
    """
    A chatbot that uses AI-driven techniques to handle user queries dynamically.

    Features:
    - Query expansion to improve response accuracy.
    - Fuzzy matching for partial or close queries.
    - Sentiment detection to adjust responses based on user emotions.
    - Context tracking for improved conversational flow.
    """

    def __init__(self, data_path):
        """
        Initializes the chatbot with a knowledge base and AI capabilities.

        Parameters:
        -----------
        data_path : str
            Path to the knowledge base data file.
        """
        self.knowledge_base = KnowledgeBase(data_path)
        self.ai_module = AIModule()

        # Context tracking for multi-step conversations
        self.conversation_history = []

        # Default fallback responses
        self.default_responses = [
            "I'm not sure about that. Can you provide more details?",
            "That's an interesting question. Let me think...",
            "I'm here to help, but I might need more information to give you a better answer."
        ]

    def handle_query(self, query):
        """
        Processes the user's query and generates a response.

        Features:
        - Expands query using synonyms for better understanding.
        - Matches user query with knowledge base using fuzzy search.
        - Adjusts responses based on detected sentiment.
        - Handles nested queries recursively.

        Parameters:
        -----------
        query : str
            User's input query.

        Returns:
        --------
        str
            Chatbot's response.
        """
        self.conversation_history.append(("User", query))

        # Normalize and expand query
        normalized_query = query.lower().strip()
        expanded_query = self.ai_module.expand_query(normalized_query)

        # Handle nested queries
        if normalized_query.startswith("nested"):
            return self.handle_nested_query(query)

        # Retrieve an exact or fuzzy-matched answer from the knowledge base
        response = self.knowledge_base.get_answer(expanded_query) or self.ai_module.fuzzy_match(expanded_query, self.knowledge_base.data)

        # If no response is found, generate a fallback response
        if not response:
            response = self.generate_response(query)

        # Detect sentiment and adjust response tone if necessary
        sentiment = self.ai_module.detect_sentiment(query)
        if sentiment == "negative":
            response += " I understand this might be frustrating. Let me try to help."

        self.conversation_history.append(("AIChatbot", response))
        return response

    def handle_nested_query(self, query):
        """
        Handles nested queries by breaking them into multiple subqueries.

        Parameters:
        -----------
        query : str
            The nested user query.

        Returns:
        --------
        str
            Combined response for all subqueries.
        """
        subqueries = query.replace("nested", "").strip().split(" and ")
        responses = [self.handle_query(subquery.strip()) for subquery in subqueries]

        return " | ".join(responses)

    def generate_response(self, query):
        """
        Generates a response for unknown queries.

        Parameters:
        -----------
        query : str
            The user query.

        Returns:
        --------
        str
            Generated response.
        """
        if "time" in query.lower():
            return f"The current time is {datetime.now().strftime('%H:%M:%S')} on {datetime.now().strftime('%Y-%m-%d')}."
        elif "date" in query.lower():
            return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        else:
            return random.choice(self.default_responses)

    def interact(self, queries):
        """
        Simulates a conversation by processing a list of queries.

        Parameters:
        -----------
        queries : list of str
            List of user queries to process.
        """
        print("AIChatbot: Hello! How can I assist you today?")
        for query in queries:
            print(f"User: {query}")
            response = self.handle_query(query)
            print(f"AIChatbot: {response}")


# Test the chatbot
if __name__ == "__main__":
    chatbot = AIChatbot("data/dev-v2.0.json")

    test_queries = [
        "What is AI?",
        "Explain dynamic programming.",
        "Tell me about recursion.",
        "How does a greedy algorithm work?",
        "What is machine learning?",
        "nested Explain natural language processing and neural networks",
        "Tell me the time.",
        "Tell me the date.",
        "Who is the president of the USA?"
    ]

    chatbot.interact(test_queries)
