import sys
import os
import pytest

# Assuming a valid path to a sample knowledge base file (adjust if needed)
SAMPLE_DATA_PATH = "data/dev-v2.0.json"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chatbot.chatbot import AIChatbot


class TestAIChatbot:
    def setup_method(self):
        """Initialize chatbot before each test."""
        self.chatbot = AIChatbot(SAMPLE_DATA_PATH)

    def test_empty_query(self):
        """Test that empty input returns a prompt."""
        result = self.chatbot.handle_query(" ")
        assert result == "Please ask me a question!"

    def test_direct_match_query(self):
        """Test if exact question from knowledge base returns expected response."""
        query = "What are two basic primary resources used to guage complexity?"
        result = self.chatbot.handle_query(query)
        assert "time" in result.lower() and "storage" in result.lower()

    def test_fuzzy_match_query(self):
        """Test if fuzzy matching retrieves related answer."""
        query = "What’s guaging complexity in computer science?"
        result = self.chatbot.handle_query(query)
        assert "Note: I found a similar question" in result or "I don't have specific information" not in result

    def test_generate_response_time(self):
        """Test fallback for time query."""
        query = "Can you tell me the time?"
        result = self.chatbot.generate_response(query)
        assert "time" in result.lower()

    def test_generate_response_date(self):
        """Test fallback for date query."""
        query = "What's the date today?"
        result = self.chatbot.generate_response(query)
        assert "date" in result.lower() or "today" in result.lower()

    def test_greeting_response(self):
        """Test for a casual greeting."""
        query = "Hello!"
        result = self.chatbot.generate_response(query)
        assert "hello" in result.lower() or "help" in result.lower()

    def test_nested_query_response(self):
        """Test compound query handling."""
        query = "What is AI and how does machine learning work?"
        result = self.chatbot.handle_query(query)
        assert "Part 1" in result and "Part 2" in result and "Answer" in result
