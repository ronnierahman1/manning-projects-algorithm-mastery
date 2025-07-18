import unittest
from chatbot.chatbot import AIChatbot
from milestones.recursive_handling import RecursiveHandling
from milestones.dynamic_context import DynamicContext
from milestones.greedy_priority import GreedyPriority

class TestChatbot(unittest.TestCase):
    """
    Test suite for the AI Chatbot, covering:
    - Standard knowledge base queries
    - Recursive handling of nested queries
    - Context-aware responses using Dynamic Programming
    - Query prioritization using a Greedy Algorithm
    """

    @classmethod
    def setUpClass(cls):
        """Initialize chatbot and modules before running tests."""
        cls.chatbot = AIChatbot("data/dev-v2.0.json")
        cls.recursive_handler = RecursiveHandling(cls.chatbot)
        cls.dynamic_context = DynamicContext()
        cls.greedy_priority = GreedyPriority()

    def test_standard_queries(self):
        """Test chatbot responses to standard knowledge base queries."""
        test_cases = {
            "What is AI?": "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines that are programmed to think and learn.",
            "Explain dynamic programming.": "Dynamic Programming is an optimization technique used to solve problems by breaking them down into simpler subproblems and storing the results for future use.",
            "Tell me about recursion.": "Recursion is a method of solving problems where a function calls itself as a subroutine to solve a smaller instance of the problem.",
            "What is machine learning?": "Machine Learning is a subset of AI that involves training algorithms to learn patterns from data and make decisions or predictions."
        }
        for query, expected_response in test_cases.items():
            with self.subTest(query=query):
                response = self.chatbot.handle_query(query)
                self.assertEqual(response, expected_response)

    def test_recursive_query_handling(self):
        """Test recursive handling of nested queries."""
        query = "nested Explain recursion and dynamic programming"
        response = self.recursive_handler.handle_recursive_query(query)
        self.assertIn("Recursion is a method", response)
        self.assertIn("Dynamic Programming is an optimization technique", response)

    def test_dynamic_context_caching(self):
        """Test if chatbot remembers previous queries for context-aware responses."""
        self.chatbot.handle_query("What is AI?")
        response = self.dynamic_context.retrieve_from_cache("What is AI?")
        self.assertEqual(response, "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines that are programmed to think and learn.")

    def test_greedy_priority_handling(self):
        """Test priority-based query optimization using the Greedy Algorithm."""
        query_high_priority = "urgent: Explain recursion"
        query_low_priority = "general: Tell me about trees"

        priority_high = self.greedy_priority.get_priority(query_high_priority)
        priority_low = self.greedy_priority.get_priority(query_low_priority)

        self.assertLess(priority_high, priority_low)  # Urgent should have a lower value (higher priority)

    def test_fuzzy_matching(self):
        """Test chatbot's ability to match partial or slightly incorrect queries."""
        test_cases = {
            "AI definition": "AI stands for Artificial Intelligence",
            "Tell me recursion": "Recursion is a method of solving problems",
            "What is ML": "Machine Learning is a subset of AI",
        }
        for query, expected_response in test_cases.items():
            with self.subTest(query=query):
                response = self.chatbot.handle_query(query)
                self.assertIn(expected_response, response)

    def test_date_time_responses(self):
        """Test chatbot responses for date and time queries."""
        response_time = self.chatbot.handle_query("What is the time?")
        response_date = self.chatbot.handle_query("What is the date?")
        self.assertIn("The current time is", response_time)
        self.assertIn("Today's date is", response_date)

    def test_unknown_query(self):
        """Test chatbot response when asked an unknown query."""
        response = self.chatbot.handle_query("Who invented pizza?")
        self.assertIn("I'm not sure about that", response)

if __name__ == "__main__":
    print("Starting chatbot tests...")
    unittest.main()
    print("Finished running chatbot tests.")
