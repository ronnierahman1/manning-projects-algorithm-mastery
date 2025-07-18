"""
Milestone 1: Test File for Recursive Query Handling

This test file verifies the behavior of the RecursiveHandling class.
It uses dummy dependencies to isolate logic from external modules and focuses
on recursive decomposition, sub-query handling, and safe recursion limits.

All external files like chatbot, knowledge_base, or main are intentionally avoided.

Learners should use this test to validate their recursive handling implementation.
"""

import pytest
from milestones.recursive_handling import RecursiveHandling


# Dummy classes for isolated testing
class DummyKnowledgeBase:
    """
    A mock KnowledgeBase used to simulate answers.
    This helps focus the test solely on the recursive logic.
    """
    def get_answer(self, query):
        return f"Answer to '{query}'"


class DummyChatbot:
    """
    A minimal chatbot mock that exposes only the expected .knowledge_base interface.
    """
    def __init__(self):
        self.knowledge_base = DummyKnowledgeBase()


@pytest.fixture
def recursive_handler():
    """
    Pytest fixture to initialize RecursiveHandling with dummy dependencies.
    """
    return RecursiveHandling(DummyChatbot())


class TestRecursiveHandling:
    def test_single_query_handling(self, recursive_handler):
        """Test a basic single query returns a direct answer."""
        query = "What is recursion?"
        response = recursive_handler.handle_recursive_query(query)
        assert "Answer to" in response

    def test_nested_query_handling(self, recursive_handler):
        """Test recursive breakdown of a query with multiple parts."""
        query = "Explain AI and ML"
        response = recursive_handler.handle_recursive_query(query)
        assert "**Question 1**" in response and "**Question 2**" in response

    def test_exceeding_recursion_depth(self, recursive_handler):
        """Ensure graceful handling when recursion depth is too high."""
        result = recursive_handler.handle_recursive_query("Part A and B", depth=10)
        assert "too complex" in result.lower()

    def test_split_simple_and(self):
        """Test that splitting logic correctly divides queries on 'and'."""
        rh = RecursiveHandling(DummyChatbot())
        parts = rh.split_into_subqueries("What is AI and what is ML")
        assert len(parts) == 2 and "AI" in parts[0] and "ML" in parts[1]

    def test_is_nested_query_true(self):
        """Confirm nested queries are detected."""
        rh = RecursiveHandling(DummyChatbot())
        assert rh._is_nested_query("Tell me about cats and dogs")

    def test_is_nested_query_false(self):
        """Confirm simple queries are not flagged as nested."""
        rh = RecursiveHandling(DummyChatbot())
        assert not rh._is_nested_query("What is physics?")
