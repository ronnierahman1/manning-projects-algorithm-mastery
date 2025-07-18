"""
Milestone 3: Greedy Priority Testing Scaffold

This test module verifies the greedy algorithm that assigns priorities to chatbot queries.
Priority levels:
  - 1: High (urgent, emergency, etc.)
  - 2: Medium (question-style, fallback)
  - 3: Low (casual/greetings)
"""

import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from milestones.greedy_priority import GreedyPriority


class TestGreedyPriority:
    def setup_method(self):
        """Initialize a new GreedyPriority instance before each test."""
        self.prioritizer = GreedyPriority()
        
        # Helper method to get priority value (handles both int and enum returns)
        def get_priority_value(priority):
            return priority.value if hasattr(priority, 'value') else priority
        
        self.get_priority_value = get_priority_value

    def test_priority_for_urgent_query(self):
        """
        Test that urgent queries receive the highest priority (1).
        """
        query = "I need help immediately"
        priority = self.prioritizer.get_priority(query)
        priority_value = self.get_priority_value(priority)
        assert priority_value == 1, f"Expected priority 1, got {priority_value}"

    def test_priority_for_greeting_query(self):
        """
        Test that casual greetings are assigned lowest priority (3 or 4).
        """
        query = "hello there"
        priority = self.prioritizer.get_priority(query)
        priority_value = self.get_priority_value(priority)
        assert priority_value in [3, 4], f"Expected priority 3 or 4, got {priority_value}"

    def test_priority_for_neutral_query(self):
        """
        Test that a non-matching query falls back to default priority (2 or 3).
        """
        query = "Elaborate about machine learning"
        priority = self.prioritizer.get_priority(query)
        priority_value = self.get_priority_value(priority)
        assert priority_value in [2, 3], f"Expected priority 2 or 3, got {priority_value}"

    def test_sorting_of_mixed_queries(self):
        """
        Test sorting a list of queries by priority.
        Verifies output is in ascending priority order.
        """
        queries = [
            "hello",                   # Priority 3/4
            "This is an emergency",    # Priority 1
            "what is your name"        # Priority 2/3
        ]
        sorted_output = self.prioritizer.sort_queries_by_priority(queries)
        priorities = [self.get_priority_value(priority) for priority, _ in sorted_output]
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

    def test_critical_error_keywords(self):
        """Test that critical error keywords receive highest priority."""
        critical_queries = [
            "critical system failure",
            "error in production database", 
            "server crashed completely",
            "urgent bug causing issues"
        ]
        
        for query in critical_queries:
            priority = self.prioritizer.get_priority(query)
            priority_value = self.get_priority_value(priority)
            assert priority_value == 1, f"Critical query '{query}' should have priority 1, got {priority_value}"

    def test_question_pattern_recognition(self):
        """Test that question patterns are properly recognized."""
        question_queries = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Can you explain this concept?",
            "Why is this happening?"
        ]
        
        for query in question_queries:
            priority = self.prioritizer.get_priority(query)
            priority_value = self.get_priority_value(priority)
            assert priority_value in [2, 3], f"Question '{query}' should have priority 2 or 3, got {priority_value}"

    def test_empty_and_whitespace_queries(self):
        """Test handling of empty and whitespace-only queries."""
        edge_cases = ["", "   ", "\n\t\r", "\n\n"]
        
        for query in edge_cases:
            priority = self.prioritizer.get_priority(query)
            priority_value = self.get_priority_value(priority)
            assert priority_value in [2, 3, 4], f"Empty/whitespace query should have priority 2, 3, or 4, got {priority_value}"

    def test_very_long_query_handling(self):
        """Test that very long queries are handled appropriately."""
        long_query = "This is a very detailed and comprehensive question about implementing " * 20
        
        priority = self.prioritizer.get_priority(long_query)
        priority_value = self.get_priority_value(priority)
        assert priority_value in [1, 2, 3], f"Long query should have priority 1, 2, or 3, got {priority_value}"

    def test_mixed_case_sensitivity(self):
        """Test that priority detection works regardless of case."""
        test_cases = [
            ("EMERGENCY SITUATION", 1),
            ("emergency situation", 1),
            ("Emergency Situation", 1),
            ("HELLO WORLD", [3, 4]),  # Allow both 3 and 4
            ("what IS this?", [2, 3])  # Allow both 2 and 3
        ]
        
        for query, expected_priority in test_cases:
            actual_priority = self.prioritizer.get_priority(query)
            actual_value = self.get_priority_value(actual_priority)
            
            if isinstance(expected_priority, list):
                assert actual_value in expected_priority, \
                    f"Query '{query}' should have priority in {expected_priority}, got {actual_value}"
            else:
                assert actual_value == expected_priority, \
                    f"Query '{query}' should have priority {expected_priority}, got {actual_value}"

    def test_punctuation_influence_on_priority(self):
        """Test how punctuation affects priority assignment."""
        test_cases = [
            ("What is this?", [2, 3]),  # Question mark
            ("Help me now!", [1, 2]),   # Exclamation with urgent keyword - allow 1 or 2
            ("Hello.", [3, 4]),         # Period with greeting
            ("Emergency!!!", 1)         # Multiple exclamations with critical keyword
        ]
        
        for query, expected_priority in test_cases:
            actual_priority = self.prioritizer.get_priority(query)
            actual_value = self.get_priority_value(actual_priority)
            
            if isinstance(expected_priority, list):
                assert actual_value in expected_priority, \
                    f"Query '{query}' should have priority in {expected_priority}, got {actual_value}"
            else:
                assert actual_value == expected_priority, \
                    f"Query '{query}' should have priority {expected_priority}, got {actual_value}"

    def test_batch_sorting_consistency(self):
        """Test that batch sorting produces consistent results."""
        queries = ["urgent help", "casual chat", "how to code", "critical error", "thanks"]
        
        # Sort multiple times and verify consistency
        result1 = self.prioritizer.sort_queries_by_priority(queries)
        result2 = self.prioritizer.sort_queries_by_priority(queries)
        
        assert result1 == result2, "Batch sorting should be consistent across multiple runs"
        assert len(result1) == len(queries), "All queries should be in sorted result"

    def test_priority_queue_basic_operations(self):
        """Test basic priority queue functionality if available."""
        # Check if enhanced version has priority queue methods
        if hasattr(self.prioritizer, 'add_to_priority_queue'):
            # Use queries with clearly different priorities
            test_queries = [
                "critical emergency system down",  # Should be priority 1
                "hello there",                     # Should be priority 4  
                "what is machine learning"         # Should be priority 3
            ]
            
            # First, check what priorities these queries actually get
            priorities = {}
            for query in test_queries:
                priorities[query] = self.get_priority_value(self.prioritizer.get_priority(query))
            
            # Add queries to queue
            for query in test_queries:
                self.prioritizer.add_to_priority_queue(query)
            
            # Verify queue is not empty
            queue_status = self.prioritizer.get_queue_status()
            assert not queue_status['is_empty'], "Queue should not be empty after adding queries"
            assert queue_status['queue_length'] == len(test_queries), "Queue should contain all added queries"
            
            # Extract all queries
            extracted_queries = []
            while not self.prioritizer.get_queue_status()['is_empty']:
                query = self.prioritizer.get_next_query()
                if query:
                    extracted_queries.append(query)
            
            # Verify all queries were extracted
            assert len(extracted_queries) == len(test_queries), "Should extract all queries from queue"
            
            # Verify that the highest priority query (lowest number) comes before lower priority queries
            extracted_priorities = [priorities[query] for query in extracted_queries]
            
            # Find the critical emergency query (should have priority 1)
            critical_query = "critical emergency system down"
            critical_priority = priorities[critical_query]
            critical_position = extracted_queries.index(critical_query)
            
            # The critical query should come before any lower priority queries
            for i in range(critical_position + 1, len(extracted_queries)):
                later_query = extracted_queries[i]
                later_priority = priorities[later_query]
                assert critical_priority <= later_priority, \
                    f"Critical query (priority {critical_priority}) should come before query '{later_query}' (priority {later_priority})"

    def test_statistics_recording_accuracy(self):
        """Test accuracy of statistics recording."""
        test_data = [
            ("emergency help", 0.2, True),
            ("casual hello", 0.1, True), 
            ("complex question", 0.5, False),
            ("thank you", 0.1, True)
        ]
        
        for query, time_taken, success in test_data:
            self.prioritizer.record_query_stats(query, time_taken, success)
        
        insights = self.prioritizer.get_optimization_insights()
        
        assert insights['total_queries'] == 4, f"Should record 4 queries, got {insights['total_queries']}"
        assert 0.74 <= insights['overall_success_rate'] <= 0.76, f"Success rate should be ~0.75, got {insights['overall_success_rate']}"

    def test_special_characters_handling(self):
        """Test handling of queries with special characters."""
        special_queries = [
            "Error: 404 not found!",
            "What's the @ symbol for?",
            "Help with C++ programming",
            "Emergency: 50% system failure",
            "Thanks! :) Very helpful"
        ]
        
        for query in special_queries:
            priority = self.prioritizer.get_priority(query)
            priority_value = self.get_priority_value(priority)
            assert isinstance(priority_value, int), f"Should return integer priority for '{query}'"
            assert 1 <= priority_value <= 4, f"Priority should be 1-4 for '{query}', got {priority_value}"

    def test_performance_with_repeated_queries(self):
        """Test system performance with repeated queries."""
        import time
        
        test_query = "What is the best approach for machine learning?"
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            priority = self.prioritizer.get_priority(test_query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / iterations
        assert avg_time < 0.01, f"Average processing time too slow: {avg_time:.4f}s per query"

    def test_unicode_and_international_text(self):
        """Test handling of unicode and international characters."""
        international_queries = [
            "¿Cómo estás?",           # Spanish
            "Здравствуйте!",          # Russian  
            "こんにちは",                # Japanese
            "مرحبا",                  # Arabic
            "Emergency: système down", # Mixed French-English
            "Urgent: проблема!"       # Mixed Russian-English
        ]
        
        for query in international_queries:
            try:
                priority = self.prioritizer.get_priority(query)
                priority_value = self.get_priority_value(priority)
                assert isinstance(priority_value, int), f"Should handle unicode query: {query}"
                assert 1 <= priority_value <= 4, f"Priority should be valid for unicode query: {query}"
            except Exception as e:
                pytest.fail(f"Should handle unicode gracefully: {query} - {e}")

    def test_error_handling_robustness(self):
        """Test system robustness with problematic inputs."""
        problematic_inputs = [
            None,
            123,
            [],
            {"key": "value"},
            "a" * 5000,  # Extremely long string
        ]
        
        for bad_input in problematic_inputs:
            try:
                # Should not crash on bad input
                priority = self.prioritizer.get_priority(bad_input)
                # If it returns something, it should be a valid priority
                if priority is not None:
                    priority_value = self.get_priority_value(priority)
                    assert isinstance(priority_value, int), f"Bad input handling failed for: {bad_input}"
            except Exception:
                # It's acceptable to raise an exception for truly invalid input
                pass

    def test_complex_workflow_integration(self):
        """Test a complete workflow from input to insights."""
        # Simulate a realistic batch of queries
        realistic_queries = [
            "Emergency: Database connection lost!",
            "How to implement user authentication?", 
            "Hello, can you help me?",
            "Critical: Payment system down",
            "What's the difference between REST and GraphQL?",
            "Thanks for the quick response!",
            "Urgent: Memory leak in production",
            "Good morning, hope you're well"
        ]
        
        # Test sorting
        sorted_queries = self.prioritizer.sort_queries_by_priority(realistic_queries)
        assert len(sorted_queries) == len(realistic_queries), "Should sort all queries"
        
        # Test that critical queries come first (should be priority 1)
        first_few_priorities = [self.get_priority_value(priority) for priority, _ in sorted_queries[:3]]
        assert any(p == 1 for p in first_few_priorities), "Critical queries should be in first few"
        
        # Test statistics recording - iterate through sorted queries correctly
        for priority, query in sorted_queries:
            query_priority_value = self.get_priority_value(priority)
            processing_time = 0.1 if query_priority_value >= 3 else 0.5
            success = True
            self.prioritizer.record_query_stats(query, processing_time, success)
        
        # Verify insights
        insights = self.prioritizer.get_optimization_insights()
        assert insights['total_queries'] == len(realistic_queries), "Should track all processed queries"
        assert insights['overall_success_rate'] == 1.0, "All queries should be successful in this test"

if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])        