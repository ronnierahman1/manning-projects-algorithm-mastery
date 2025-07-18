"""
Milestone 3: Greedy Priority Testing Scaffold

This test module verifies the greedy algorithm that assigns priorities to chatbot queries.
Priority levels:
  - 1: High (urgent, emergency, etc.)
  - 2: Medium (question-style, fallback)
  - 3: Low (casual/greetings)
"""

import pytest
from milestones.greedy_priority import GreedyPriority


class TestGreedyPriority:
    def setup_method(self):
        """Initialize a new GreedyPriority instance before each test."""
        self.prioritizer = GreedyPriority()

    def test_priority_for_urgent_query(self):
        """
        Test that urgent queries receive the highest priority (1).
        """
        query = "I need help immediately"
        priority = self.prioritizer.get_priority(query)
        assert priority == 1, f"Expected priority 1, got {priority}"

    def test_priority_for_greeting_query(self):
        """
        Test that casual greetings are assigned lowest priority (3).
        """
        query = "hello there"
        priority = self.prioritizer.get_priority(query)
        assert priority == 3, f"Expected priority 3, got {priority}"

    def test_priority_for_neutral_query(self):
        """
        Test that a non-matching query falls back to default priority (3).
        """
        query = "Elaborate about machine learning"
        priority = self.prioritizer.get_priority(query)
        assert priority == 3, f"Expected priority 3, got {priority}"

    def test_sorting_of_mixed_queries(self):
        """
        Test sorting a list of queries by priority.
        Verifies output is in ascending priority order.
        """
        queries = [
            "hello",                   # Priority 3
            "This is an emergency",    # Priority 1
            "what is your name"        # Priority 2
        ]
        sorted_output = self.prioritizer.sort_queries_by_priority(queries)
        priorities = [priority for priority, _ in sorted_output]
        assert priorities == sorted(priorities), "Queries not sorted correctly by priority"

    def test_optional_optimization_insights(self):
        """
        Test that get_optimization_insights returns a valid stats dictionary after recording.
        """
        import random

        queries = [
            "emergency now",
            "hello world",
            "explain AI",
            "thank you"
        ]

        for q in queries:
            self.prioritizer.record_query_stats(
                query=q,
                processing_time=round(random.uniform(0.1, 0.5), 3),
                success=random.choice([True, False])
            )

        insights = self.prioritizer.get_optimization_insights()
        assert isinstance(insights, dict), "Insights should return a dictionary"
        assert "total_queries" in insights, "Expected 'total_queries' in insights"
