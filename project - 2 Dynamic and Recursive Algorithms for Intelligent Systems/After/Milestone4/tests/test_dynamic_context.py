"""
Enhanced Test File for Dynamic Context Caching

This comprehensive test file verifies the behavior of the DynamicContext class
with extensive coverage of core functionality, edge cases, and advanced features.
"""

import pytest
import datetime
import os, sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from milestones.dynamic_context import DynamicContext


class TestDynamicContext:
    def setup_method(self):
        """
        Run before each test method to initialize a clean context.
        """
        self.context = DynamicContext()

    def test_store_and_retrieve_exact_match(self):
        """
        Test basic cache insertion and retrieval using the same query.
        """
        query = "What is a graph?"
        response = "A graph is a collection of nodes and edges."

        self.context.store_in_cache(query, response)
        result = self.context.retrieve_from_cache(query)

        assert result == response, "Exact match retrieval failed."

    def test_cache_is_normalized(self):
        """
        Test that the query is normalized (e.g., case/punctuation) during retrieval.
        """
        query = "What is a Graph?"
        response = "A graph is a collection of nodes and edges."

        self.context.store_in_cache(query, response)

        # Test using slightly varied casing and punctuation
        result = self.context.retrieve_from_cache("what is a graph")
        assert result == response, "Normalization during cache lookup failed."

    def test_cache_miss_returns_none(self):
        """
        Ensure that a query with no match returns None.
        """
        query = "Explain dynamic programming"
        result = self.context.retrieve_from_cache(query)

        assert result is None, "Unexpected result for missing cache entry."

    def test_context_window_limit(self):
        """
        Ensure context history does not exceed the configured context window size.
        """
        for i in range(15):
            self.context.store_in_cache(f"Question {i}", f"Answer {i}")

        history = self.context.conversation_context
        assert len(history) <= self.context.context_window * 2, "Context window exceeded allowed limit."

    def test_fuzzy_cache_match_scaffold(self):
        """
        (Advanced) Fuzzy matching based on word overlap should retrieve the best approximate match.
        """
        self.context.store_in_cache("Define recursion", "Recursion is when a function calls itself.")
        self.context.store_in_cache("What is sorting?", "Sorting is arranging items in order.")

        # TODO: This should match approximately using fuzzy logic
        result = self.context.retrieve_from_cache("definition of recursive method")

        # ✅ Learners: Modify the fuzzy matching threshold if needed
        assert result == "Recursion is when a function calls itself." or result is None

    def test_has_context_true(self):
        """
        Verify that an inserted query is correctly detected in the cache.
        """
        query = "What is dynamic programming?"
        self.context.store_in_cache(query, "It's an optimization technique.")

        assert self.context.has_context(query) is True

    def test_has_context_false(self):
        """
        Verify that unrelated queries are not falsely detected as cached.
        """
        query = "What is memoization?"
        assert self.context.has_context(query) is False

    def test_analyze_context_followup(self):
        """
        Test whether a follow-up word is detected in a query.
        """
        self.context.store_in_cache("What is recursion?", "Recursion is a self-calling process.")
        self.context.store_in_cache("Tell me more about it", "It means solving smaller subproblems.")

        analysis = self.context._analyze_context("What about sorting algorithms?")

        assert analysis["is_followup"] is True
        assert isinstance(analysis["related_topics"], list)

    def test_cache_access_count_increment(self):
        """
        Verify that cache access count is properly incremented on retrieval.
        """
        query = "What is binary search?"
        response = "Binary search is a divide-and-conquer algorithm."
        
        self.context.store_in_cache(query, response)
        
        # Initial access count should be 1
        cache_key = self.context._normalize_query(query)
        assert self.context.response_cache[cache_key]['access_count'] == 1
        
        # Retrieve multiple times and check access count
        self.context.retrieve_from_cache(query)
        self.context.retrieve_from_cache(query)
        
        assert self.context.response_cache[cache_key]['access_count'] == 3

    def test_cache_timestamp_storage(self):
        """
        Verify that timestamps are properly stored and updated.
        """
        query = "What is a hash table?"
        response = "A hash table is a data structure that maps keys to values."
        
        with patch('datetime.datetime') as mock_datetime:
            mock_now = datetime.datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = mock_now
            
            self.context.store_in_cache(query, response)
            
            cache_key = self.context._normalize_query(query)
            assert self.context.response_cache[cache_key]['timestamp'] == mock_now

    def test_conversation_context_structure(self):
        """
        Verify that conversation context maintains proper structure with user/bot entries.
        """
        query = "How does quicksort work?"
        response = "Quicksort uses divide-and-conquer with a pivot element."
        
        self.context.store_in_cache(query, response)
        
        # Should have 2 entries (user + bot)
        assert len(self.context.conversation_context) == 2
        
        # Check user entry
        user_entry = self.context.conversation_context[0]
        assert user_entry['type'] == 'user'
        assert user_entry['content'] == query
        assert 'timestamp' in user_entry
        
        # Check bot entry
        bot_entry = self.context.conversation_context[1]
        assert bot_entry['type'] == 'bot'
        assert bot_entry['content'] == response
        assert 'timestamp' in bot_entry

    def test_fuzzy_matching_high_threshold(self):
        """
        Test fuzzy matching with various similarity scores.
        """
        self.context.store_in_cache("machine learning algorithms", "ML algorithms are used for pattern recognition.")
        self.context.store_in_cache("data structures", "Data structures organize data efficiently.")
        
        # High similarity should match
        result = self.context.retrieve_from_cache("machine learning methods")
        assert result == "ML algorithms are used for pattern recognition."
        
        # Low similarity should not match
        result = self.context.retrieve_from_cache("cooking recipes")
        assert result is None

    def test_fuzzy_matching_empty_words(self):
        """
        Test fuzzy matching edge case with empty word sets.
        """
        self.context.store_in_cache("", "Empty query response")
        
        result = self.context.retrieve_from_cache("")
        assert result == "Empty query response"

    def test_analyze_context_related_topics(self):
        """
        Test that related topics are properly extracted from conversation history.
        """
        # Add some conversation history
        self.context.store_in_cache("What is machine learning?", "ML is a subset of AI.")
        self.context.store_in_cache("Tell me about neural networks", "Neural networks are inspired by the brain.")
        
        analysis = self.context._analyze_context("How do algorithms work?")
        
        # Should extract meaningful words from recent queries
        assert 'machine' in analysis['related_topics'] or 'learning' in analysis['related_topics']
        assert 'neural' in analysis['related_topics'] or 'networks' in analysis['related_topics']

    def test_analyze_context_no_followup(self):
        """
        Test that queries without follow-up indicators are correctly identified.
        """
        analysis = self.context._analyze_context("What is Python programming?")
        
        assert analysis['is_followup'] is False
        assert analysis['sentiment_shift'] == 'neutral'

    def test_normalize_query_variations(self):
        """
        Test query normalization with various punctuation and casing.
        """
        test_cases = [
            ("What is AI?", "what is ai"),
            ("MACHINE LEARNING!", "machine learning"),
            ("  Data Science  ", "data science"),
            ("How does this work???", "how does this work")
        ]
        
        for original, expected in test_cases:
            normalized = self.context._normalize_query(original)
            assert normalized == expected

    def test_error_handling_in_store_cache(self):
        """
        Test error handling when storing cache entries.
        """
        with patch.object(self.context, '_normalize_query', side_effect=Exception("Test error")):
            # Should not raise exception
            self.context.store_in_cache("test query", "test response")
            
            # Cache should remain empty
            assert len(self.context.response_cache) == 0

    def test_error_handling_in_retrieve_cache(self):
        """
        Test error handling when retrieving from cache.
        """
        with patch.object(self.context, '_normalize_query', side_effect=Exception("Test error")):
            result = self.context.retrieve_from_cache("test query")
            assert result is None

    def test_error_handling_in_has_context(self):
        """
        Test error handling in has_context method.
        """
        with patch.object(self.context, '_normalize_query', side_effect=Exception("Test error")):
            result = self.context.has_context("test query")
            assert result is False

    def test_multiple_cache_entries_fuzzy_best_match(self):
        """
        Test that fuzzy matching returns the best match when multiple similar entries exist.
        """
        self.context.store_in_cache("python programming language", "Python is a high-level language.")
        self.context.store_in_cache("java programming tutorial", "Java is an object-oriented language.")
        self.context.store_in_cache("python data analysis", "Python is great for data analysis.")
        
        # Should match the most similar entry
        result = self.context.retrieve_from_cache("python programming basics")
        assert result == "Python is a high-level language."


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

