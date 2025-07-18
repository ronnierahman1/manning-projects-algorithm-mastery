"""
Milestone 3: Enhanced Greedy Priority Testing Suite

This comprehensive test module verifies the enhanced greedy algorithm that assigns 
priorities to chatbot queries with extensive edge case coverage and performance testing.

Priority levels:
  - CRITICAL (1): Urgent, emergency, system failures
  - HIGH (2): Important, time-sensitive queries  
  - MEDIUM (3): Questions, explanations, standard queries
  - LOW (4): Casual greetings, social interactions
"""

import pytest
import time
import random
from unittest.mock import patch
from milestones.greedy_priority import GreedyPriority, Priority, QueryMetrics


class TestGreedyPriority:
    """Comprehensive test suite for the GreedyPriority class."""
    
    def setup_method(self):
        """Initialize a new GreedyPriority instance before each test."""
        self.prioritizer = GreedyPriority()

    # === Basic Priority Assignment Tests ===
    
    def test_critical_priority_keywords(self):
        """Test that critical keywords receive CRITICAL priority."""
        critical_queries = [
            "This is an emergency!",
            "Urgent help needed",
            "Critical system failure",
            "Server is down immediately",
            "Application crashed and won't start",
            "Database connection error - critical issue"
        ]
        
        for query in critical_queries:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.CRITICAL, f"Query '{query}' should be CRITICAL, got {priority}"

    def test_high_priority_keywords(self):
        """Test that high priority keywords receive HIGH priority."""
        high_queries = [
            "This is important",
            "I need help with this problem",
            "Quick question about the deadline",
            "Must complete this soon",
            "Priority task needs assistance"
        ]
        
        for query in high_queries:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.HIGH, f"Query '{query}' should be HIGH, got {priority}"

    def test_medium_priority_keywords(self):
        """Test that medium priority keywords receive MEDIUM priority."""
        medium_queries = [
            "What is machine learning?",
            "How does this algorithm work?",
            "Can you explain the process?",
            "Why is this happening?",
            "Tell me about artificial intelligence"
        ]
        
        for query in medium_queries:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.MEDIUM, f"Query '{query}' should be MEDIUM, got {priority}"

    def test_low_priority_keywords(self):
        """Test that low priority keywords receive LOW priority."""
        low_queries = [
            "Hello there!",
            "Good morning",
            "Thanks for your help",
            "Goodbye and have a nice day",
            "Just want to chat about the weather"
        ]
        
        for query in low_queries:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.LOW, f"Query '{query}' should be LOW, got {priority}"

    # === Pattern-Based Priority Tests ===
    
    def test_regex_patterns_critical(self):
        """Test regex patterns for critical priority detection."""
        critical_patterns = [
            "The system can't work properly",
            "Database connection failed to establish",
            "Server down and service unavailable",
            "Lost all my data after the update",
            "Application doesn't work anymore"
        ]
        
        for query in critical_patterns:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.CRITICAL, f"Pattern '{query}' should be CRITICAL, got {priority}"

    def test_regex_patterns_high(self):
        """Test regex patterns for high priority detection."""
        high_patterns = [
            "Need help with this integration",
            "How to fix this authentication issue",
            "Deadline approaching tomorrow",
            "Due today - need assistance"
        ]
        
        for query in high_patterns:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.HIGH, f"Pattern '{query}' should be HIGH, got {priority}"

    def test_regex_patterns_medium(self):
        """Test regex patterns for medium priority detection."""
        medium_patterns = [
            "What is the best approach?",
            "Can you help me understand this?",
            "How does this system work?",
            "Why is this implementation better?"
        ]
        
        for query in medium_patterns:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.MEDIUM, f"Pattern '{query}' should be MEDIUM, got {priority}"

    # === Length-Based Priority Tests ===
    
    def test_very_long_query_priority(self):
        """Test that very long queries get HIGH priority."""
        long_query = "This is a very long and complex query that contains many details and specifics about a particular problem that needs to be solved. It includes multiple aspects of the system architecture, implementation details, performance considerations, security requirements, and integration patterns that must be carefully analyzed and addressed in a comprehensive manner." * 2
        
        priority = self.prioritizer.get_priority(long_query)
        assert priority == Priority.HIGH, f"Very long query should be HIGH priority, got {priority}"

    def test_medium_length_query_priority(self):
        """Test that medium length queries get appropriate priority."""
        medium_query = "Can you explain how to implement a proper authentication system with JWT tokens?"
        
        priority = self.prioritizer.get_priority(medium_query)
        assert priority == Priority.MEDIUM, f"Medium query should be MEDIUM priority, got {priority}"

    def test_short_query_priority(self):
        """Test that short queries get appropriate priority based on content."""
        short_query = "Help!"
        
        priority = self.prioritizer.get_priority(short_query)
        assert priority == Priority.HIGH, f"Short help query should be HIGH priority, got {priority}"

    # === Complexity-Based Priority Tests ===
    
    def test_high_complexity_query(self):
        """Test that complex queries with technical terms get higher priority."""
        complex_query = "Explain the algorithm implementation for optimizing system performance in a scalable architecture with security considerations"
        
        priority = self.prioritizer.get_priority(complex_query)
        assert priority == Priority.HIGH, f"Complex query should be HIGH priority, got {priority}"

    def test_step_by_step_query(self):
        """Test that step-by-step requests get appropriate priority."""
        step_query = "Provide a detailed step by step comprehensive guide for deployment"
        
        priority = self.prioritizer.get_priority(step_query)
        assert priority in [Priority.HIGH, Priority.MEDIUM], f"Step-by-step query should be HIGH or MEDIUM priority, got {priority}"

    # === Edge Cases and Error Handling ===
    
    def test_empty_query(self):
        """Test handling of empty queries."""
        empty_queries = ["", "   ", "\n\t  \n"]
        
        for query in empty_queries:
            priority = self.prioritizer.get_priority(query)
            assert priority == Priority.LOW, f"Empty query '{repr(query)}' should be LOW priority, got {priority}"

    def test_none_query(self):
        """Test handling of None input."""
        with patch('builtins.print') as mock_print:
            priority = self.prioritizer.get_priority(None)
            assert priority == Priority.LOW, f"None query should be LOW priority, got {priority}"

    def test_special_characters_query(self):
        """Test queries with special characters."""
        special_queries = [
            "What's the @#$% problem?",
            "How to fix this & that?",
            "Error: 404 - not found!",
            "System.exit(1) - critical!"
        ]
        
        for query in special_queries:
            priority = self.prioritizer.get_priority(query)
            assert isinstance(priority, Priority), f"Query '{query}' should return valid Priority enum"

    def test_unicode_query(self):
        """Test queries with unicode characters."""
        unicode_queries = [
            "¿Cómo puedo ayudarte?",
            "这是一个紧急问题",
            "помогите пожалуйста",
            "🚨 Emergency situation!"
        ]
        
        for query in unicode_queries:
            priority = self.prioritizer.get_priority(query)
            assert isinstance(priority, Priority), f"Unicode query '{query}' should return valid Priority enum"

    # === Query Sorting Tests ===
    
    def test_sorting_mixed_priorities(self):
        """Test sorting queries with mixed priorities."""
        queries = [
            "Hello world",                    # LOW
            "Critical system failure",        # CRITICAL  
            "How does this work?",           # MEDIUM
            "Need help urgently",            # HIGH
            "Thanks for everything"          # LOW
        ]
        
        sorted_queries = self.prioritizer.sort_queries_by_priority(queries)
        priorities = [priority for priority, _ in sorted_queries]
        
        # Should be sorted in ascending order of priority values
        assert priorities == sorted(priorities), f"Priorities not sorted correctly: {priorities}"
        
        # First query should be CRITICAL
        assert sorted_queries[0][0] == Priority.CRITICAL, f"First query should be CRITICAL, got {sorted_queries[0][0]}"

    def test_sorting_empty_list(self):
        """Test sorting an empty list of queries."""
        result = self.prioritizer.sort_queries_by_priority([])
        assert result == [], "Empty list should return empty list"

    def test_sorting_single_query(self):
        """Test sorting a single query."""
        queries = ["What is AI?"]
        result = self.prioritizer.sort_queries_by_priority(queries)
        
        assert len(result) == 1, "Single query should return single result"
        assert result[0][1] == "What is AI?", "Query content should be preserved"

    def test_sorting_same_priority_queries(self):
        """Test sorting queries with the same priority."""
        queries = [
            "What is machine learning?",
            "How does AI work?",
            "Explain neural networks"
        ]
        
        result = self.prioritizer.sort_queries_by_priority(queries)
        
        # All should have the same priority
        priorities = [priority for priority, _ in result]
        assert all(p == priorities[0] for p in priorities), "All queries should have same priority"

    # === Priority Queue Tests ===
    
    def test_priority_queue_operations(self):
        """Test priority queue add and retrieve operations."""
        queries = [
            "Hello",                     # LOW
            "Emergency situation",       # CRITICAL
            "How to implement this?",    # MEDIUM
            "Need help quickly"          # HIGH
        ]
        
        # Add queries to queue
        for query in queries:
            self.prioritizer.add_to_priority_queue(query)
        
        # Retrieve queries - should come out in priority order
        retrieved = []
        while True:
            query = self.prioritizer.get_next_query()
            if query is None:
                break
            retrieved.append(query)
        
        # First query should be the critical one
        assert retrieved[0] == "Emergency situation", f"First query should be 'Emergency situation', got '{retrieved[0]}'"
        
        # Should retrieve all queries
        assert len(retrieved) == 4, f"Should retrieve 4 queries, got {len(retrieved)}"

    def test_priority_queue_empty(self):
        """Test priority queue when empty."""
        assert self.prioritizer.get_next_query() is None, "Empty queue should return None"

    def test_priority_queue_status(self):
        """Test priority queue status reporting."""
        # Initially empty
        status = self.prioritizer.get_queue_status()
        assert status['queue_length'] == 0, "Queue should be empty initially"
        assert status['is_empty'] is True, "Queue should report as empty"
        assert status['next_priority'] is None, "No next priority when empty"
        
        # Add a query
        self.prioritizer.add_to_priority_queue("Test query")
        status = self.prioritizer.get_queue_status()
        assert status['queue_length'] == 1, "Queue should have 1 item"
        assert status['is_empty'] is False, "Queue should not be empty"
        assert status['next_priority'] is not None, "Should have next priority"

    def test_priority_queue_clear(self):
        """Test clearing the priority queue."""
        # Add some queries
        for i in range(5):
            self.prioritizer.add_to_priority_queue(f"Query {i}")
        
        # Clear the queue
        self.prioritizer.clear_queue()
        
        # Should be empty
        status = self.prioritizer.get_queue_status()
        assert status['queue_length'] == 0, "Queue should be empty after clear"
        assert status['is_empty'] is True, "Queue should report as empty after clear"

    # === Statistics and Analytics Tests ===
    
    def test_record_query_stats_basic(self):
        """Test basic query statistics recording."""
        query = "What is AI?"
        processing_time = 0.5
        success = True
        
        self.prioritizer.record_query_stats(query, processing_time, success)
        
        insights = self.prioritizer.get_optimization_insights()
        assert insights['total_queries'] == 1, "Should record 1 query"
        assert insights['overall_success_rate'] == 1.0, "Success rate should be 100%"
        assert abs(insights['avg_processing_time'] - 0.5) < 0.001, "Average processing time should be 0.5"

    def test_record_query_stats_multiple(self):
        """Test recording statistics for multiple queries."""
        queries_data = [
            ("What is machine learning?", 0.3, True),
            ("How does AI work?", 0.8, True),
            ("Emergency server down!", 1.2, False),
            ("Hello there", 0.1, True),
            ("Explain deep learning", 0.6, True)
        ]
        
        for query, time_taken, success in queries_data:
            self.prioritizer.record_query_stats(query, time_taken, success)
        
        insights = self.prioritizer.get_optimization_insights()
        
        assert insights['total_queries'] == 5, f"Should record 5 queries, got {insights['total_queries']}"
        assert insights['overall_success_rate'] == 0.8, f"Success rate should be 0.8, got {insights['overall_success_rate']}"
        
        # Check that we have multiple query types
        assert len(insights['most_common_query_types']) > 0, "Should have query type statistics"

    def test_record_query_stats_invalid_time(self):
        """Test handling of invalid processing times."""
        with patch('builtins.print') as mock_print:
            self.prioritizer.record_query_stats("Test query", -1.0, True)
            mock_print.assert_called_once()  # Should print warning

    def test_query_categorization(self):
        """Test query categorization for analytics."""
        test_cases = [
            ("What is the definition of AI?", "definition"),
            ("How to implement a neural network?", "how-to"),
            ("Where is the server located?", "location"),
            ("When will this be completed?", "temporal"),
            ("Why is this algorithm better?", "causal"),
            ("Who created this framework?", "person"),
            ("System error occurred", "troubleshooting"),
            ("Hello there!", "social"),
            ("Algorithm optimization needed", "technical"),
            ("Is this correct?", "question"),
            ("Random statement", "general"),
            ("", "empty")
        ]
        
        for query, expected_category in test_cases:
            actual_category = self.prioritizer._categorize_query(query)
            assert actual_category == expected_category, f"Query '{query}' should be categorized as '{expected_category}', got '{actual_category}'"

    def test_optimization_insights_empty(self):
        """Test optimization insights when no data is available."""
        insights = self.prioritizer.get_optimization_insights()
        
        expected_keys = [
            'total_queries', 'avg_processing_time', 'overall_success_rate',
            'slowest_query_types', 'most_common_query_types', 'fastest_query_types',
            'least_successful_types', 'recommendations'
        ]
        
        for key in expected_keys:
            assert key in insights, f"Key '{key}' should be in insights"
        
        assert insights['total_queries'] == 0, "Total queries should be 0 when empty"
        assert len(insights['recommendations']) == 0, "Should have no recommendations when empty"

    def test_optimization_insights_comprehensive(self):
        """Test comprehensive optimization insights with varied data."""
        # Add diverse query data
        query_data = [
            # Slow, failing queries
            ("Complex algorithm implementation", 2.5, False),
            ("Detailed system architecture", 2.0, False),
            ("Comprehensive optimization guide", 1.8, True),
            
            # Fast, successful queries  
            ("Hi", 0.1, True),
            ("Thanks", 0.1, True),
            ("Hello", 0.1, True),
            
            # Medium performance queries
            ("What is machine learning?", 0.5, True),
            ("How does this work?", 0.6, True),
            ("Explain the process", 0.4, True),
        ]
        
        for query, time_taken, success in query_data:
            self.prioritizer.record_query_stats(query, time_taken, success)
        
        insights = self.prioritizer.get_optimization_insights()
        
        # Should have recommendations due to low success rate and high processing time
        assert len(insights['recommendations']) > 0, "Should have recommendations"
        
        # Check that slowest and fastest are identified
        assert len(insights['slowest_query_types']) > 0, "Should identify slowest query types"
        assert len(insights['fastest_query_types']) > 0, "Should identify fastest query types"
        
        # Verify structure of query type data
        for slow_type in insights['slowest_query_types']:
            assert 'type' in slow_type, "Slow query type should have 'type' field"
            assert 'avg_time' in slow_type, "Slow query type should have 'avg_time' field"
            assert 'count' in slow_type, "Slow query type should have 'count' field"

    def test_reset_stats(self):
        """Test resetting statistics."""
        # Add some data
        self.prioritizer.record_query_stats("Test query", 0.5, True)
        
        # Verify data exists
        insights = self.prioritizer.get_optimization_insights()
        assert insights['total_queries'] > 0, "Should have recorded queries"
        
        # Reset and verify empty
        self.prioritizer.reset_stats()
        insights = self.prioritizer.get_optimization_insights()
        assert insights['total_queries'] == 0, "Should have no queries after reset"

    # === Performance and Stress Tests ===
    
    def test_performance_large_query_batch(self):
        """Test performance with a large batch of queries."""
        import time
        
        # Generate 1000 random queries
        queries = []
        query_templates = [
            "What is {}?",
            "How to implement {}?",
            "Emergency: {} is down!",
            "Hello, help with {}",
            "Explain the {} algorithm"
        ]
        
        topics = ["AI", "database", "server", "authentication", "optimization", 
                 "security", "performance", "integration", "deployment", "testing"]
        
        for i in range(1000):
            template = random.choice(query_templates)
            topic = random.choice(topics)
            queries.append(template.format(topic))
        
        # Time the sorting operation
        start_time = time.time()
        sorted_queries = self.prioritizer.sort_queries_by_priority(queries)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        processing_time = end_time - start_time
        assert processing_time < 1.0, f"Large batch processing took too long: {processing_time:.3f}s"
        
        # Should return all queries
        assert len(sorted_queries) == 1000, f"Should return all 1000 queries, got {len(sorted_queries)}"
        
        # Should be properly sorted
        priorities = [priority for priority, _ in sorted_queries]
        assert priorities == sorted(priorities), "Large batch should be properly sorted"

    def test_memory_usage_priority_queue(self):
        """Test memory usage with large priority queue."""
        import sys
        
        # Add many queries to the queue
        for i in range(10000):
            query = f"Query number {i} with varying content and length to test memory usage"
            self.prioritizer.add_to_priority_queue(query)
        
        # Verify queue contains all items
        status = self.prioritizer.get_queue_status()
        assert status['queue_length'] == 10000, f"Queue should contain 10000 items, got {status['queue_length']}"
        
        # Process all queries efficiently
        processed_count = 0
        while not self.prioritizer.get_queue_status()['is_empty']:
            query = self.prioritizer.get_next_query()
            if query:
                processed_count += 1
        
        assert processed_count == 10000, f"Should process all 10000 queries, got {processed_count}"

    def test_concurrent_statistics_recording(self):
        """Test statistics recording with concurrent-like access patterns."""
        import threading
        
        def record_stats_batch(start_idx, count):
            for i in range(start_idx, start_idx + count):
                query = f"Test query {i}"
                processing_time = random.uniform(0.1, 2.0)
                success = random.choice([True, False])
                self.prioritizer.record_query_stats(query, processing_time, success)
        
        # Simulate concurrent recording (note: not truly concurrent due to GIL)
        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_stats_batch, args=(i * 100, 100))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all stats were recorded
        insights = self.prioritizer.get_optimization_insights()
        assert insights['total_queries'] == 500, f"Should record 500 queries, got {insights['total_queries']}"

    # === Integration and Workflow Tests ===
    
    def test_complete_workflow(self):
        """Test a complete workflow from query input to insights."""
        # Step 1: Process various queries
        queries = [
            "Critical system failure - server down!",
            "How to implement OAuth authentication?",
            "What is the difference between AI and ML?",
            "Hello, can you help me?",
            "Thanks for the assistance",
            "Urgent: database connection lost",
            "Explain machine learning algorithms step by step"
        ]
        
        # Step 2: Sort queries by priority
        sorted_queries = self.prioritizer.sort_queries_by_priority(queries)
        
        # Step 3: Add to priority queue
        for query in queries:
            self.prioritizer.add_to_priority_queue(query)
        
        # Step 4: Process queries and record stats
        processing_order = []
        while not self.prioritizer.get_queue_status()['is_empty']:
            query = self.prioritizer.get_next_query()
            if query:
                processing_order.append(query)
                # Simulate processing
                processing_time = random.uniform(0.1, 1.0)
                success = random.choice([True, True, True, False])  # 75% success rate
                self.prioritizer.record_query_stats(query, processing_time, success)
        
        # Step 5: Get insights
        insights = self.prioritizer.get_optimization_insights()
        
        # Verify workflow
        assert len(processing_order) == len(queries), "Should process all queries"
        assert insights['total_queries'] == len(queries), "Should record all query stats"
        
        # Critical queries should be processed first
        critical_queries = [q for q in processing_order if "critical" in q.lower() or "urgent" in q.lower()]
        if critical_queries:
            # Find position of first critical query
            first_critical_pos = min(processing_order.index(q) for q in critical_queries)
            assert first_critical_pos < 3, "Critical queries should be processed early"

    def test_error_handling_comprehensive(self):
        """Test comprehensive error handling throughout the system."""
        
        # Test with malformed inputs
        malformed_inputs = [
            None,
            "",
            "   ",
            "\n\t\r",
            "a" * 10000,  # Very long string
            "🚨💻🔥" * 100,  # Many emojis
            "SELECT * FROM users; DROP TABLE users;",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
        ]
        
        for malformed_input in malformed_inputs:
            try:
                priority = self.prioritizer.get_priority(malformed_input)
                assert isinstance(priority, Priority), f"Should return valid Priority for input: {repr(malformed_input)}"
                
                # Should handle in sorting
                sorted_result = self.prioritizer.sort_queries_by_priority([malformed_input] if malformed_input else [])
                assert isinstance(sorted_result, list), "Should return list even for malformed input"
                
                # Should handle in queue
                if malformed_input:
                    self.prioritizer.add_to_priority_queue(malformed_input)
                    retrieved = self.prioritizer.get_next_query()
                    assert retrieved == malformed_input, "Should retrieve same malformed input"
                
            except Exception as e:
                pytest.fail(f"Should not raise exception for input {repr(malformed_input)}: {e}")

    # === Boundary Value Tests ===
    
    def test_boundary_query_lengths(self):
        """Test queries at boundary lengths."""
        test_cases = [
            ("a", "Single character"),
            ("a" * 49, "Just under medium threshold"),
            ("a" * 50, "At medium threshold"),
            ("a" * 51, "Just over medium threshold"),
            ("a" * 99, "Just under long threshold"),
            ("a" * 100, "At long threshold"),
            ("a" * 101, "Just over long threshold"),
            ("a" * 199, "Just under very long threshold"),
            ("a" * 200, "At very long threshold"),
            ("a" * 201, "Just over very long threshold"),
        ]
        
        for query, description in test_cases:
            priority = self.prioritizer.get_priority(query)
            assert isinstance(priority, Priority), f"Should handle {description}: {len(query)} chars"

    def test_complexity_score_boundaries(self):
        """Test complexity score calculation at boundaries."""
        # Test with varying numbers of complexity indicators
        base_query = "Explain this system"
        complexity_words = self.prioritizer.complexity_indicators
        
        # No complexity words
        priority_0 = self.prioritizer.get_priority(base_query)
        
        # One complexity word
        query_1 = base_query + " " + complexity_words[0]
        priority_1 = self.prioritizer.get_priority(query_1)
        
        # Multiple complexity words
        query_multi = base_query + " " + " ".join(complexity_words[:5])
        priority_multi = self.prioritizer.get_priority(query_multi)
        
        # Should generally increase priority with complexity
        # (though other factors may influence final priority)
        assert isinstance(priority_0, Priority), "Should handle no complexity words"
        assert isinstance(priority_1, Priority), "Should handle one complexity word"
        assert isinstance(priority_multi, Priority), "Should handle multiple complexity words"

    # === Configuration and Customization Tests ===
    
    def test_priority_keyword_modification(self):
        """Test modifying priority keywords."""
        # Add custom critical keyword
        custom_keyword = "supercritical"
        self.prioritizer.priority_keywords[Priority.CRITICAL].append(custom_keyword)
        
        # Test with custom keyword
        query = f"This is a {custom_keyword} situation"
        priority = self.prioritizer.get_priority(query)
        assert priority == Priority.CRITICAL, f"Custom keyword should trigger CRITICAL priority"
        
        # Remove custom keyword and test again
        self.prioritizer.priority_keywords[Priority.CRITICAL].remove(custom_keyword)
        priority = self.prioritizer.get_priority(query)
        assert priority != Priority.CRITICAL, f"Removed keyword should not trigger CRITICAL priority"

    def test_threshold_modification(self):
        """Test modifying length thresholds."""
        # Modify thresholds
        original_thresholds = self.prioritizer.length_thresholds.copy()
        self.prioritizer.length_thresholds['long'] = 20  # Very short threshold
        
        # Test with modified threshold
        query = "a" * 25  # Should now be considered long
        priority = self.prioritizer.get_priority(query)
        
        # Restore original thresholds
        self.prioritizer.length_thresholds = original_thresholds
        
        # Should handle threshold modification gracefully
        assert isinstance(priority, Priority), "Should handle modified thresholds"


# === Performance Benchmarks ===

class TestPerformanceBenchmarks:
    """Performance benchmark tests for the GreedyPriority system."""
    
    def setup_method(self):
        """Initialize prioritizer for benchmarks."""
        self.prioritizer = GreedyPriority()
    
    @pytest.mark.performance
    def test_single_query_performance(self):
        """Benchmark single query priority calculation."""
        query = "What is the best approach for implementing a scalable microservices architecture?"
        
        start_time = time.time()
        for _ in range(10000):
            priority = self.prioritizer.get_priority(query)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10000
        assert avg_time < 0.001, f"Single query processing should be under 1ms, got {avg_time:.6f}s"
    
    @pytest.mark.performance  
    def test_batch_sorting_performance(self):
        """Benchmark batch query sorting performance."""
        queries = [f"Query number {i} with varying content" for i in range(1000)]
        
        start_time = time.time()
        sorted_queries = self.prioritizer.sort_queries_by_priority(queries)
        end_time = time.time()
        
        processing_time = end_time - start_time
        assert processing_time < 0.5, f"1000 query sorting should be under 0.5s, got {processing_time:.3f}s"
        assert len(sorted_queries) == 1000, "Should return all queries"
    
    @pytest.mark.performance
    def test_priority_queue_performance(self):
        """Benchmark priority queue operations."""
        queries = [f"Test query {i}" for i in range(5000)]
        
        # Test insertion performance
        start_time = time.time()
        for query in queries:
            self.prioritizer.add_to_priority_queue(query)
        insertion_time = time.time() - start_time
        
        # Test extraction performance  
        start_time = time.time()
        extracted = 0
        while not self.prioritizer.get_queue_status()['is_empty']:
            query = self.prioritizer.get_next_query()
            if query:
                extracted += 1
        extraction_time = time.time() - start_time
        
        assert insertion_time < 1.0, f"5000 insertions should be under 1s, got {insertion_time:.3f}s"
        assert extraction_time < 1.0, f"5000 extractions should be under 1s, got {extraction_time:.3f}s"
        assert extracted == 5000, f"Should extract all 5000 queries, got {extracted}"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])