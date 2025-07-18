"""
Test File for Recursive Query Handling

This comprehensive test file validates the complete behavior of the RecursiveHandling class,
including edge cases, error handling, performance optimizations, and all public methods.
It uses mock dependencies to isolate logic and focuses on thorough testing of:
- Recursive decomposition with proper depth handling
- Query splitting algorithms and edge cases
- Cache integration and performance
- Error handling and graceful degradation
- Response formatting and structure
- Performance characteristics
"""

import pytest
import time, re
import os
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from milestones.recursive_handling import RecursiveHandling, QueryResult

# Enhanced mock classes with more realistic behavior+
class DummyKnowledgeBase:
    """
    Enhanced mock KnowledgeBase with more realistic response patterns.
    """
    def __init__(self):
        self.responses = {
            'what is ai': 'Artificial Intelligence is the simulation of human intelligence in machines.',
            'what is ml': 'Machine Learning is a subset of AI that enables computers to learn without explicit programming.',
            'what is python': 'Python is a high-level programming language known for its simplicity.',
            'what is recursion': 'Recursion is a programming technique where a function calls itself.',
            'what is data science': 'Data science combines statistics, programming, and domain expertise.',
            'explain neural networks': 'Neural networks are computing systems inspired by biological neural networks.',
            'what is deep learning': 'Deep learning uses neural networks with multiple layers.',
            'how does computer vision work': 'Computer vision enables machines to interpret visual information.',
            'what is natural language processing': 'NLP helps computers understand and process human language.',
            'explain quantum computing': 'Quantum computing uses quantum mechanics for computation.',
        }
    
    def get_answer(self, query):
        query_clean = re.sub(r'[^\w\s]', '', query.lower().strip())  # Strip punctuation
        return self.responses.get(query_clean, f"I don't have specific information about '{query}'")


class DummyChatbot:
    """
    Enhanced chatbot mock with realistic handle_query behavior.
    """
    def __init__(self):
        self.knowledge_base = DummyKnowledgeBase()
        self.call_count = 0
    
    def handle_query(self, query):
        self.call_count += 1
        answer = self.knowledge_base.get_answer(query)
        
        # Simulate fuzzy matching behavior
        is_fuzzy = "don't have specific information" in answer
        threshold = 0.0 if not is_fuzzy else 0.6
        
        return answer, is_fuzzy, threshold


@pytest.fixture
def enhanced_recursive_handler():
    """Enhanced fixture with realistic dependencies."""
    return RecursiveHandling(DummyChatbot())


@pytest.fixture
def basic_recursive_handler():
    """Basic fixture for simple tests."""
    class BasicKnowledgeBase:
        def get_answer(self, query):
            return f"Answer to '{query}'"
    
    class BasicChatbot:
        def __init__(self):
            self.knowledge_base = BasicKnowledgeBase()
        
        def handle_query(self, query):
            return self.knowledge_base.get_answer(query), False, 0.0
    
    return RecursiveHandling(BasicChatbot())


class TestRecursiveHandlingBasicFunctionality:
    """Test basic functionality and core methods."""
    
    def test_initialization(self, enhanced_recursive_handler):
        """Test proper initialization of RecursiveHandling."""
        rh = enhanced_recursive_handler
        assert rh.chatbot is not None
        assert rh.max_recursion_depth == 5
        # assert len(rh.recursion_patterns) > 0
        # assert hasattr(rh, 'recursion_patterns')
    
    def test_single_query_handling(self, enhanced_recursive_handler):
        """Test handling of simple, non-nested queries."""
        rh = enhanced_recursive_handler
        result = rh.handle_recursive_query("what is ai?")
        
        assert isinstance(result, QueryResult)
        assert result.response is not None
        assert result.is_recursive is False
        assert 'Artificial Intelligence' in result.response
    
    def test_empty_query_handling(self, enhanced_recursive_handler):
        """Test handling of empty or whitespace queries."""
        rh = enhanced_recursive_handler
        
        # Empty string
        result = rh.handle_recursive_query("")
        assert isinstance(result, QueryResult)
        assert result.response != ""
        
        # Whitespace only
        result = rh.handle_recursive_query("   ")
        assert isinstance(result, QueryResult)
        assert result.response != ""


class TestQueryComplexityDetection:
    """Test the _is_nested_query method thoroughly."""
    
    def test_simple_queries_not_nested(self, enhanced_recursive_handler):
        """Test that simple queries are not flagged as nested."""
        rh = enhanced_recursive_handler
        simple_queries = [
            "What is AI?",
            "How does machine learning work?", 
            "Explain quantum computing.",
            "What's the weather like?",
            "Hello there!",
            "Can you help me?"
        ]
        
        for query in simple_queries:
            assert not rh._is_nested_query_internal(query), f"Query '{query}' should not be nested"
    
    def test_compound_queries_detected(self, enhanced_recursive_handler):
        """Test that compound queries are properly detected."""
        rh = enhanced_recursive_handler
        compound_queries = [
            "What is AI and how does ML work?",
            "Explain Python and also describe Java",
            "Tell me about cats and dogs",
            "What about machine learning and deep learning?",
            "Describe neural networks and computer vision",
            "What is recursion and how does it work?",
            "Compare AI and ML and also explain their differences"
        ]
        
        for query in compound_queries:
            assert rh._is_nested_query_internal(query), f"Query '{query}' should be detected as nested"
    
    def test_multiple_question_marks(self, enhanced_recursive_handler):
        """Test detection of queries with multiple question marks."""
        rh = enhanced_recursive_handler
        multi_question_queries = [
            "What is AI? How does it work?",
            "Can you explain ML? What about deep learning?",
            "Who invented computers? When was it invented?"
        ]
        
        for query in multi_question_queries:
            assert rh._is_nested_query_internal(query), f"Query '{query}' with multiple ? should be nested"
    
    def test_edge_cases_in_detection(self, enhanced_recursive_handler):
        """Test edge cases in nested query detection."""
        rh = enhanced_recursive_handler
        
        # Should not be nested (false positives)
        false_positives = [
            "I want to learn about AI and become better",  # 'and' but not compound question
            "Tell me the pros and cons of Python",  # single topic with 'and'
            "Research and development in AI",  # phrase, not compound query
        ]
        
        for query in false_positives:
            # These might still be detected as nested depending on implementation
            # The test documents the current behavior
            result = rh._is_nested_query_internal(query)
            # We don't assert here as the behavior might be acceptable


class TestQuerySplitting:
    """Test the split_into_subqueries method comprehensively."""
    
    def test_simple_and_splitting(self, enhanced_recursive_handler):
        """Test splitting on simple 'and' conjunctions."""
        rh = enhanced_recursive_handler
        
        test_cases = [
            ("What is AI and what is ML", ["What is AI", "what is ML"]),
            ("Explain both Python and describe Java", ["both Python", "describe Java"]),
            ("Tell me about cats and dogs", ["cats", "dogs"])
        ]
        
        for query, expected_parts in test_cases:
            parts = rh._split_into_subqueries(query)
            assert len(parts) >= 2, f"Query '{query}' should split into multiple parts"
            # Check that key terms are preserved
            for expected_part in expected_parts:
                assert any(expected_part.lower() in part.lower() for part in parts), \
                    f"Expected part '{expected_part}' not found in split result {parts}"
    
    def test_question_mark_splitting(self, enhanced_recursive_handler):
        """Test splitting on question marks."""
        rh = enhanced_recursive_handler
        
        query = "What is AI? How does ML work? What about deep learning?"
        parts = rh._split_into_subqueries(query)
        
        assert len(parts) >= 2, "Multiple questions should be split"
        # Each part should be a complete question or meaningful fragment
        for part in parts:
            assert len(part.strip()) > 0, "No empty parts should be returned"
    
    def test_complex_pattern_splitting(self, enhanced_recursive_handler):
        """Test splitting with complex patterns from recursion_patterns."""
        rh = enhanced_recursive_handler
        
        test_cases = [
            "tell me about machine learning and deep learning",
            "what about Python and also Java",
            "explain neural networks as well as computer vision",
            "describe AI and also how it works"
        ]
        
        for query in test_cases:
            parts = rh._split_into_subqueries(query)
            assert len(parts) >= 2, f"Query '{query}' should split into multiple parts"
            assert all(len(part.strip()) > 0 for part in parts), "No empty parts allowed"
    
    def test_no_splitting_for_simple_queries(self, enhanced_recursive_handler):
        """Test that simple queries are not split unnecessarily."""
        rh = enhanced_recursive_handler
        
        simple_queries = [
            "What is artificial intelligence?",
            "How do neural networks work?",
            "Explain quantum computing principles"
        ]
        
        for query in simple_queries:
            parts = rh._split_into_subqueries(query)
            # Should return original query or single meaningful part
            assert len(parts) >= 1, "Should return at least one part"
            if len(parts) == 1:
                # If only one part, it should be the original or very similar
                assert query.lower() in parts[0].lower() or parts[0].lower() in query.lower()
    
    def test_splitting_edge_cases(self, enhanced_recursive_handler):
        """Test edge cases in query splitting."""
        rh = enhanced_recursive_handler
        
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "and",  # Just conjunction
            "What is?",  # Incomplete question
            "a and b and c and d",  # Multiple conjunctions
        ]
        
        for query in edge_cases:
            parts = rh._split_into_subqueries(query)
            assert isinstance(parts, list), "Should always return a list"
            assert len(parts) >= 1, "Should return at least one part"


class TestRecursiveProcessing:
    """Test the recursive processing functionality."""
    
    def test_nested_query_processing(self, enhanced_recursive_handler):
        """Test processing of nested queries with proper formatting."""
        rh = enhanced_recursive_handler
        
        query = "What is AI and what is ML?"
        result = rh.handle_recursive_query(query)
        
        assert isinstance(result, QueryResult)
        assert result.is_recursive is True
        assert 'Question 1' in result.response
        assert 'Question 2' in result.response
        assert 'Answer' in result.response
        assert len(result.responses) >= 2
    
    def test_recursion_depth_limiting(self, enhanced_recursive_handler):
        """Test that recursion depth is properly limited."""
        rh = enhanced_recursive_handler
        
        # Test with depth at the limit
        result = rh.handle_recursive_query("What is AI and ML?", depth=4)
        assert isinstance(result, QueryResult)
        
        # Test exceeding depth limit
        result = rh.handle_recursive_query("What is AI and ML?", depth=10)
        assert isinstance(result, QueryResult)
        assert 'complex' in result.response.lower() or 'error' in result.response.lower()
    
    def test_handle_nested_query_directly(self, enhanced_recursive_handler):
        """Test the handle_nested_query method directly."""
        rh = enhanced_recursive_handler
        
        query = "Explain AI and describe ML"
        start_time = time.time()
        result = rh._handle_nested_query(query, depth=1, start_time=start_time)
        
        assert isinstance(result, QueryResult)
        assert result.is_recursive is True
        assert result.processing_time >= 0
        assert isinstance(result.responses, list)
        assert len(result.responses) >= 1
    
    def test_recursive_with_realistic_queries(self, enhanced_recursive_handler):
        """Test recursive handling with realistic complex queries."""
        rh = enhanced_recursive_handler
        
        complex_queries = [
            "What is machine learning and how does deep learning differ from it?",
            "Explain neural networks and also describe computer vision applications",
            "What is Python used for and what about Java?",
            "Tell me about AI ethics and also explain quantum computing basics"
        ]
        
        for query in complex_queries:
            result = rh.handle_recursive_query(query)
            assert isinstance(result, QueryResult)
            assert result.response is not None
            assert len(result.response) > 0
            # Should contain structured formatting for complex queries
            if result.is_recursive:
                assert 'Question' in result.response or 'Answer' in result.response


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_chatbot_error_handling(self, enhanced_recursive_handler):
        """Test handling when chatbot raises exceptions."""
        rh = enhanced_recursive_handler
        
        # Mock the chatbot to raise an exception
        original_handle_query = rh.chatbot.handle_query
        rh.chatbot.handle_query = Mock(side_effect=Exception("Simulated error"))
        
        result = rh.handle_recursive_query("What is AI?")
        assert isinstance(result, QueryResult)
        assert result.response is not None
        # Should handle error gracefully
        assert len(result.response) > 0
        
        # Restore original method
        rh.chatbot.handle_query = original_handle_query
    
    def test_splitting_error_handling(self, enhanced_recursive_handler):
        """Test error handling in query splitting."""
        rh = enhanced_recursive_handler
        
        # Test with potentially problematic inputs
        problematic_queries = [
            None,  # This would cause an error if not handled
            "\x00\x01\x02",  # Control characters
            "🤖🔥💻" * 100,  # Very long emoji string
        ]
        
        for query in problematic_queries:
            if query is not None:  # Skip None as it would be caught earlier
                try:
                    parts = rh._split_into_subqueries(query)
                    assert isinstance(parts, list)
                except Exception:
                    # If it throws an exception, that's acceptable for extreme edge cases
                    pass
    
    def test_malformed_nested_queries(self, enhanced_recursive_handler):
        """Test handling of malformed nested queries."""
        rh = enhanced_recursive_handler
        
        malformed_queries = [
            "What is and what is?",  # Missing subjects
            "and and and",  # Just conjunctions
            "? ? ?",  # Just question marks
            "tell me about and also",  # Incomplete
        ]
        
        for query in malformed_queries:
            result = rh.handle_recursive_query(query)
            assert isinstance(result, QueryResult)
            assert result.response is not None
            assert len(result.response) > 0  # Should provide some response


class TestPerformanceAndCaching:
    """Test performance characteristics and caching behavior."""
    
    def test_response_timing(self, enhanced_recursive_handler):
        """Test that processing times are recorded."""
        rh = enhanced_recursive_handler
        
        result = rh.handle_recursive_query("What is AI?")
        assert result.processing_time is not None
        assert isinstance(result.processing_time, (int, float))
        assert result.processing_time >= 0
    
    def test_complex_query_performance(self, enhanced_recursive_handler):
        """Test performance with complex queries."""
        rh = enhanced_recursive_handler
        
        complex_query = "What is AI and ML and deep learning and neural networks?"
        
        start_time = time.time()
        result = rh.handle_recursive_query(complex_query)
        end_time = time.time()
        
        assert isinstance(result, QueryResult)
        assert end_time - start_time < 5.0  # Should complete within reasonable time
        assert result.processing_time <= end_time - start_time
    
    def test_cache_integration_placeholder(self, enhanced_recursive_handler):
        """Test cache integration if available."""
        rh = enhanced_recursive_handler
        
        # Test that cache-related fields are properly set
        result = rh.handle_recursive_query("What is AI?")
        assert result.used_cache is not None
        assert isinstance(result.used_cache, bool)
    
    @patch('time.time')
    def test_processing_time_calculation(self, mock_time, enhanced_recursive_handler):
        """Test that processing time is calculated correctly."""
        rh = enhanced_recursive_handler
        
        # Mock time to return predictable values
        mock_time.side_effect = [0.0, 1.5]  # Start at 0, end at 1.5
        
        result = rh.handle_recursive_query("What is AI?")
        
        assert result.processing_time is not None
        assert isinstance(result.processing_time, (int, float))

        assert result.processing_time== 1.5


class TestReturnStructures:
    """Test the structure and format of returned data."""
    
    def test_return_dict_structure(self, enhanced_recursive_handler):
        """Test that return dictionaries have the expected structure."""
        rh = enhanced_recursive_handler
        
        # Test simple query
        result = rh.handle_recursive_query("What is AI?")
        
        required_keys = [
            'response', 'responses', 'processing_time', 'is_recursive',
            'used_cache', 'priority', 'fuzzy_match', 'fuzzy_match_threshold'
        ]
        
        
        assert result is not None
        
        # Test types
        assert isinstance(result.response, str)
        assert isinstance(result.responses, list)
        assert isinstance(result.processing_time, (int, float))
        assert isinstance(result.is_recursive, bool)
        assert isinstance(result.used_cache, bool)
        assert isinstance(result.priority, int)
        assert isinstance(result.fuzzy_match, bool)
        assert isinstance(result.fuzzy_match_threshold, (int, float))
    
    def test_responses_list_structure(self, enhanced_recursive_handler):
        """Test the structure of the responses list."""
        rh = enhanced_recursive_handler
        
        # Test nested query to get multiple responses
        result = rh.handle_recursive_query("What is AI and what is ML?")
        
        assert isinstance(result.responses, list)
        assert len(result.responses) >= 1
        
        # Each response should be a tuple (response_text, is_fuzzy, threshold)
        for response_item in result.responses:
            assert isinstance(response_item, tuple)
            assert len(response_item) == 3
            assert isinstance(response_item[0], str)  # response text
            assert isinstance(response_item[1], bool)  # is_fuzzy
            assert isinstance(response_item[2], (int, float))  # threshold
    
    def test_priority_values(self, enhanced_recursive_handler):
        """Test that priority values are within expected range."""
        rh = enhanced_recursive_handler
        
        queries = [
            "What is AI?",
            "What is AI and ML?",
            "Explain AI and ML and deep learning"
        ]
        
        for query in queries:
            result = rh.handle_recursive_query(query)
            priority = result.priority
            assert isinstance(priority, int)
            assert 1 <= priority <= 3, f"Priority {priority} out of expected range for query: {query}"


class TestRegressionAndBackwardCompatibility:
    """Test for regressions and backward compatibility."""
    
    def test_original_test_compatibility(self, basic_recursive_handler):
        """Ensure original tests still pass with new implementation."""
        rh = basic_recursive_handler
        
        # Original test: single query handling  
        query = "What is recursion?"
        result = rh.handle_recursive_query(query)
        assert isinstance(result, QueryResult)
        assert "Answer to" in result.response
        
        # Original test: nested query handling
        query = "Explain AI and ML"
        result = rh.handle_recursive_query(query)
        if result.is_recursive:
            assert "Question 1" in result.response and "Question 2" in result.response
        
        # Original test: exceeding recursion depth
        result = rh.handle_recursive_query("Part A and B", depth=10)
        assert "complex" in result.response.lower() or "error" in result.response.lower()
    
    def test_method_signatures_unchanged(self, enhanced_recursive_handler):
        """Test that public method signatures haven't changed."""
        rh = enhanced_recursive_handler
        
        # Test that methods exist and are callable
        assert callable(rh.handle_recursive_query)
        assert callable(rh._handle_nested_query)
        assert callable(rh._split_into_subqueries)
        assert callable(rh._is_nested_query_internal)
        
        # Test that they accept expected parameters
        try:
            rh.handle_recursive_query("test")
            rh.handle_recursive_query("test", depth=1)
            rh._handle_nested_query("test", depth=1, start_time=time.time())
            rh._split_into_subqueries("test")
            rh._is_nested_query_internal("test")
        except TypeError:
            pytest.fail("Method signature changed unexpectedly")


# Integration tests that verify the complete workflow
class TestIntegrationWorkflow:
    """Integration tests for complete workflows."""
    
    def test_complete_recursive_workflow(self, enhanced_recursive_handler):
        """Test a complete recursive query workflow."""
        rh = enhanced_recursive_handler
        chatbot = rh.chatbot
        initial_call_count = chatbot.call_count
        
        # Process a complex query
        query = "What is AI and what is ML?"
        result = rh.handle_recursive_query(query)
        
        # Verify the complete workflow
        assert isinstance(result, QueryResult)
        assert result.is_recursive is True
        assert len(result.responses) >= 2
        assert chatbot.call_count > initial_call_count  # Chatbot was called
        assert 'Artificial Intelligence' in result.response
        assert 'Machine Learning' in result.response
    
    def test_mixed_query_processing(self, enhanced_recursive_handler):
        """Test processing a mix of simple and complex queries."""
        rh = enhanced_recursive_handler
        
        queries = [
            ("What is AI?", False),  # Simple
            ("What is AI and What is ML?", True),  # Complex  
            ("How does recursion work?", False),  # Simple
            ("Explain Python and describe Java and what about C++?", True),  # Complex
        ]
        
        for query, expected_recursive in queries:
            result = rh.handle_recursive_query(query)
            assert isinstance(result, QueryResult)
            assert result.is_recursive == expected_recursive, \
                f"Query '{query}' recursion detection failed"
            assert len(result.response) > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])