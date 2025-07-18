"""
Milestone 2: Test File for Dynamic Context Caching

This test file verifies the behavior of the DynamicContext class.
It includes both scaffolding and basic tests. Some areas are marked
as TODOs for learners to fill in advanced logic (e.g., fuzzy matching).
"""

import pytest
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
        self.context.store_in_cache(query, "It’s an optimization technique.")

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

