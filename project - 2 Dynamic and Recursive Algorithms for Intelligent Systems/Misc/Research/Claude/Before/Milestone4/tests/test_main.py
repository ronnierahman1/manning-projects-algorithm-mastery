# tests/test_main.py

import unittest
import tempfile
import json
import os
import sys

# Add the root directory to sys.path for importing main app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import RecursiveAIChatbotApp


class TestMainIntegration(unittest.TestCase):
    """Integration tests for the main RecursiveAIChatbotApp."""

    @classmethod
    def setUpClass(cls):
        # Create a temporary knowledge base file
        cls.test_data = {
            "data": [
                {
                    "title": "Sample QA",
                    "paragraphs": [
                        {
                            "context": "Artificial intelligence is a branch of computer science.",
                            "qas": [
                                {
                                    "question": "What is AI?",
                                    "answers": [{"text": "AI is the simulation of human intelligence in machines."}],
                                    "id": "1"
                                },
                                {
                                    "question": "Define AI",
                                    "answers": [{"text": "AI stands for Artificial Intelligence."}],
                                    "id": "2"
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        cls.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(cls.test_data, cls.temp_file)
        cls.temp_file.close()

        # Initialize the main app with the temporary knowledge base
        cls.app = RecursiveAIChatbotApp(data_path=cls.temp_file.name)

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.temp_file.name)

    def test_simple_query(self):
        """Test a standard query returns an expected response."""
        query = "What is AI?"
        result = self.app.process_query(query)
        self.assertTrue(result["success"])
        self.assertIn("AI", result["response"])

    def test_recursive_query(self):
        """Test a compound query triggers recursive handling."""
        query = "What is AI and Define AI"
        result = self.app.process_query(query)
        self.assertTrue(result["is_recursive"])
        self.assertIn("**Question 1**", result["response"])
        self.assertIn("**Answer**", result["response"])

    def test_cache_functionality(self):
        """Test that a response is cached and reused."""
        query = "What is AI?"
        first = self.app.process_query(query)
        second = self.app.process_query(query)
        self.assertFalse(first["used_cache"])
        self.assertTrue(second["used_cache"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
