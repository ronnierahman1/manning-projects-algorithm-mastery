"""
Optimized Recursive Query Handling - MILESTONE 1 STARTER CODE

This module implements efficient logic for identifying and processing nested or compound queries.
It breaks down complex questions into simpler sub-questions and processes them individually
to provide informative and structured responses to the user.

LEARNING OBJECTIVES:
- Understand how to implement recursive logic for query processing
- Learn to detect when queries contain multiple parts
- Practice breaking complex problems into simpler sub-problems
- Implement recursive function calls with proper depth control

KEY CONCEPTS TO IMPLEMENT:
1. Recursive query detection - identifying when a query has multiple parts
2. Query splitting - breaking complex queries into manageable pieces
3. Recursive processing - calling functions recursively to handle sub-queries
4. Result aggregation - combining multiple sub-results into a coherent response
"""

import re  # Regular expressions for pattern matching
from typing import List, Dict, Any, Optional, Tuple  # Type hints for better code documentation
from dataclasses import dataclass  # For creating structured data classes
from functools import lru_cache  # Least Recently Used cache decorator for performance
import time  # For measuring processing time


@dataclass
class QueryResult:
    """
    Data class to standardize query results
    
    This class acts as a container for all information about how a query was processed.
    Using a dataclass makes the code more readable and ensures consistent return values.
    """
    response: str  # The final response text to show the user
    responses: List[Tuple[str, bool, float]]  # List of (response_text, is_fuzzy_match, confidence_score)
    processing_time: float  # How long it took to process the query (in seconds)
    is_recursive: bool  # Whether this query was split into multiple parts
    used_cache: bool  # Whether cached results were used (for performance tracking)
    priority: int  # Priority level (1=simple, 2=complex) for response ordering
    fuzzy_match: bool  # Whether fuzzy matching was used to find answers
    fuzzy_match_threshold: float  # Confidence threshold for fuzzy matching (0.0-1.0)


class RecursiveHandling:
    """
    Optimized class to process nested (recursive) user queries by breaking them into sub-questions.
    
    IMPLEMENTATION GUIDE:
    This class needs to implement recursive logic that can:
    1. Detect when queries are compound (contain multiple parts)
    2. Split compound queries into individual sub-queries
    3. Process each sub-query recursively
    4. Combine results into a coherent response
    5. Handle edge cases and prevent infinite recursion
    """

    def __init__(self, chatbot, max_recursion_depth: int = 5):
        """
        Initialize the RecursiveHandling class.

        Args:
            chatbot: A reference to the main chatbot instance that handles individual queries
            max_recursion_depth: Maximum recursion depth to prevent infinite loops
        """
        # Store reference to the main chatbot for processing individual queries
        self.chatbot = chatbot
        # Set limit to prevent infinite recursion (safety mechanism)
        self.max_recursion_depth = max_recursion_depth
        
        # Pre-compile regex patterns for better performance
        self._compile_patterns()
        
        # Cache for nested query detection to avoid re-analyzing the same queries
        self._nested_query_cache = {}

    def _compile_patterns(self) -> None:
        """
        Pre-compile all regex patterns for better performance
        
        TODO: IMPLEMENT PATTERN COMPILATION
        You need to:
        1. Create compound query patterns that detect queries with multiple parts
        2. Create question boundary patterns to detect new questions after "and"
        3. Create punctuation splitting patterns for different sentence boundaries
        4. Compile all patterns with appropriate flags
        
        HINT: Look for patterns like:
        - "tell me about X and Y"
        - "what is X and how do I Y"
        - "X and also Y"
        """
        # STEP 1: Define compound query patterns
        # These patterns should capture two groups: the parts before and after connecting words
        compound_patterns = [
            r'tell me about (.*?) and (.*?)(?:\?|$)',  # "tell me about X and Y"
            r'what about (.*?) and (.*?)(?:\?|$)',     # "what about X and Y"
            r'explain (.*?) and (.*?)(?:\?|$)',        # "explain X and Y"
            r'describe (.*?) and (.*?)(?:\?|$)',       # "describe X and Y"
            r'(.*?) and also (.*?)(?:\?|$)',           # "X and also Y"
            r'(.*?) as well as (.*?)(?:\?|$)'          # "X as well as Y"
        ]
        
        # TODO: Compile all patterns with IGNORECASE flag
        # HINT: Use re.compile() for each pattern in the list
        # self.compound_patterns = [your code here]
        
        # STEP 2: Create question boundary pattern
        # This should detect when a new question starts after "and"
        question_words = ['what', 'how', 'why', 'where', 'when', 'who', 'which']
        
        # TODO: Create pattern that matches "and/& + question_word"
        # HINT: Use word boundaries (\b) and alternation (|) to match any question word
        # self.question_boundary_pattern = re.compile(your_pattern, re.IGNORECASE)
        
        # STEP 3: Create punctuation splitting patterns
        # TODO: Create patterns to split on different punctuation marks
        # self.question_split_pattern = re.compile(your_pattern)     # Split on question marks
        # self.sentence_split_pattern = re.compile(your_pattern)     # Split on periods and exclamation marks
        # self.clause_split_pattern = re.compile(your_pattern)       # Split on any sentence-ending punctuation

    @lru_cache(maxsize=100)  # Cache the last 100 query checks to improve performance
    def _is_nested_query_cached(self, query: str) -> bool:
        """
        Cached version of nested query detection for frequently asked queries
        
        The @lru_cache decorator automatically caches results for performance.
        """
        return self._is_nested_query_internal(query)

    def _is_nested_query_internal(self, query: str) -> bool:
        """
        Internal method to determine if a query contains multiple sub-questions.
        
        TODO: IMPLEMENT NESTED QUERY DETECTION
        You need to implement logic that detects if a query is compound using these heuristics:
        1. Check for explicit connecting words (and, also, etc.)
        2. Count question marks (multiple ? usually means multiple questions)
        3. Look for multiple question words in different sentence parts
        
        Args:
            query: The user input string
            
        Returns:
            bool: True if the query is compound/multi-part
        """
        # STEP 1: Basic validation
        # TODO: Return False if query is None, empty, or too short (less than 10 characters)
        
        # STEP 2: Convert to lowercase for case-insensitive matching
        # TODO: Create a lowercase version of the query
        
        # STEP 3: Define nested indicators
        # These words/phrases often connect multiple questions
        nested_indicators = {
            ' and ', ' & ', '; ', ' or ', ' plus ',        # Direct connectors
            'also', 'additionally', 'furthermore', 'moreover',  # Additive words
            'what about', 'how about', 'tell me about',          # Question starters
            'explain both', 'describe both', 'compare'           # Explicit multi-part requests
        }
        
        # TODO: HEURISTIC 1 - Check for explicit indicators
        # Return True if any nested indicator is found in the query
        # HINT: Use the 'in' operator to check if each indicator exists in query_lower
        
        # TODO: HEURISTIC 2 - Count question marks
        # Return True if the query contains more than one question mark
        # HINT: Use the count() method on the original query
        
        # TODO: HEURISTIC 3 - Multiple question words in different parts
        # Steps:
        # 1. Define question_words set
        # 2. Split query into parts using clause_split_pattern
        # 3. For each part, check if it contains any question words
        # 4. Return True if multiple parts contain question words
        
        question_words = {'what', 'how', 'why', 'where', 'when', 'who', 'which'}
        # TODO: Implement the logic to split and analyze parts
        
        # TODO: Return False if none of the heuristics detected a compound query

    def handle_recursive_query(self, query: str, depth: int = 0) -> QueryResult:
        """
        Primary entry point for handling recursive queries.
        
        TODO: IMPLEMENT MAIN RECURSIVE LOGIC
        This method should:
        1. Check recursion depth to prevent infinite loops
        2. Detect if query is nested (only at depth 0)
        3. Route to appropriate handler (nested vs single)
        4. Handle any errors gracefully
        
        Args:
            query: The user input string (potentially complex)
            depth: Tracks the recursion level to prevent overflow
            
        Returns:
            QueryResult: Structured result containing response and metadata
        """
        # STEP 1: Record start time for performance measurement
        start_time = time.time()
        
        try:
            # STEP 2: Guard clause for recursion depth
            # TODO: Check if depth >= max_recursion_depth
            # If so, return a QueryResult with an error message about complexity
            # HINT: Use the provided QueryResult structure with appropriate values
            
            # STEP 3: Check for nested queries (only at initial call)
            # TODO: If depth == 0 AND the query is nested (use _is_nested_query_cached):
            #       - Call _handle_nested_query with depth + 1
            #       - Return the result
            
            # STEP 4: Process as single query
            # TODO: Call _handle_single_query and return the result
            
        except Exception as e:
            # Handle any unexpected errors gracefully
            print(f"Error in handle_recursive_query: {e}")
            return QueryResult(
                response="An error occurred while processing your query. Please try rephrasing it.",
                responses=[],
                processing_time=time.time() - start_time,
                is_recursive=True,
                used_cache=False,
                priority=1,
                fuzzy_match=False,
                fuzzy_match_threshold=0.0
            )

    def _handle_single_query(self, query: str, start_time: float) -> QueryResult:
        """
        Handle a single, non-nested query
        
        TODO: IMPLEMENT SINGLE QUERY PROCESSING
        This method should:
        1. Delegate query processing to the chatbot
        2. Wrap the result in QueryResult format
        
        Args:
            query: A simple, single-part query
            start_time: When we started processing (for timing)
            
        Returns:
            QueryResult: Standardized result object
        """
        # STEP 1: Call the chatbot to handle the query
        # TODO: Call self.chatbot.handle_query(query)
        # This returns: (response_text, is_fuzzy_match, confidence_threshold)
        
        # STEP 2: Create and return QueryResult
        # TODO: Wrap the chatbot response in QueryResult format
        # HINT: Set is_recursive=False, priority=1, and use the chatbot's return values

    def _handle_nested_query(self, query: str, depth: int, start_time: float) -> QueryResult:
        """
        Handle compound queries by splitting and processing each part.
        
        TODO: IMPLEMENT NESTED QUERY PROCESSING
        This is the core of recursive processing. You need to:
        1. Split the query into sub-queries
        2. Process each sub-query recursively
        3. Combine all results into a coherent response
        4. Handle cases where splitting fails
        
        Args:
            query: A nested user query (contains multiple parts)
            depth: Current recursion depth (for safety)
            start_time: Time when processing started
            
        Returns:
            QueryResult: Combined result from all sub-queries
        """
        try:
            # STEP 1: Split the complex query into simpler sub-queries
            # TODO: Call _split_into_subqueries(query) to get a list of sub-queries
            
            # STEP 2: Check if splitting was successful
            # TODO: If sub_queries has <= 1 item, call _handle_single_query as fallback
            
            # STEP 3: Initialize storage for responses
            responses = []           # Raw responses for metadata
            formatted_responses = [] # Formatted responses for display
            
            # STEP 4: Process each sub-query
            # TODO: Loop through sub_queries with enumerate to get index and sub_query
            # For each sub_query:
            # 1. Strip whitespace and skip if empty
            # 2. Recursively call handle_recursive_query with depth + 1
            # 3. Extract response, fuzzy_match, and threshold from result
            # 4. Add to responses list as tuple
            # 5. Format for display and add to formatted_responses
            
            # STEP 5: Handle case where no valid responses were generated
            # TODO: If responses is empty, call _handle_single_query as fallback
            
            # STEP 6: Combine all responses
            # TODO: Create combined_response by joining formatted_responses with headers
            # HINT: Start with "I'll address each part of your question:\n\n"
            
            # STEP 7: Create and return final QueryResult
            # TODO: Return QueryResult with:
            # - combined_response as response
            # - responses list
            # - is_recursive=True
            # - priority=2
            # - fuzzy_match=True if ANY response used fuzzy matching
            # - fuzzy_match_threshold=max threshold from all responses

        except Exception as e:
            print(f"Error in _handle_nested_query: {e}")
            return QueryResult(
                response="I had trouble processing your complex question. Please try asking each part separately.",
                responses=[],
                processing_time=time.time() - start_time,
                is_recursive=True,
                used_cache=False,
                priority=2,
                fuzzy_match=False,
                fuzzy_match_threshold=0.0
            )

    def _split_into_subqueries(self, query: str) -> List[str]:
        """
        Optimized method to split a complex query into constituent sub-questions.
        
        TODO: IMPLEMENT QUERY SPLITTING ALGORITHM
        This method should try multiple strategies in order:
        1. Compound patterns (most reliable)
        2. Question boundary detection
        3. Punctuation-based splitting
        4. Conjunction-based splitting
        
        Args:
            query: The full query text
            
        Returns:
            List[str]: A list of simpler, self-contained queries
        """
        # STEP 1: Basic validation
        # TODO: Return [query] if query is None, empty, or less than 10 characters
        
        try:
            # STEP 2: Try compound patterns first (most reliable)
            # TODO: Loop through self.compound_patterns
            # For each pattern:
            # 1. Use pattern.search(query) to find matches
            # 2. If match found, extract groups and filter out empty ones
            # 3. If len(groups) > 1, return groups
            
            # STEP 3: Try question boundary detection
            # TODO: Call _split_by_question_boundaries(query)
            # If result has > 1 part, return it
            
            # STEP 4: Try punctuation-based splitting
            # TODO: Call _split_by_punctuation(query)
            # If result has > 1 part, return it
            
            # STEP 5: Try conjunction-based splitting
            # TODO: Call _split_by_conjunctions(query)
            # If result has > 1 part, return it
            
            # STEP 6: Return original query if no splitting worked
            return [query]

        except Exception as e:
            print(f"Error in _split_into_subqueries: {e}")
            return [query]

    def _split_by_question_boundaries(self, query: str) -> List[str]:
        """
        Split query by question word boundaries like 'and where', 'and what'
        
        TODO: IMPLEMENT QUESTION BOUNDARY SPLITTING
        Steps:
        1. Find all matches of question_boundary_pattern
        2. Split query at each boundary
        3. Return list of parts (only if multiple valid parts found)
        """
        # TODO: Use self.question_boundary_pattern.finditer(query) to find matches
        # TODO: If no matches, return [query]
        # TODO: Split query at each boundary, keeping substantial parts (length > 3)
        # TODO: Return parts if len(parts) > 1, otherwise return [query]
        return [query]  # Placeholder - replace with your implementation

    def _split_by_punctuation(self, query: str) -> List[str]:
        """
        Split query by punctuation marks
        
        TODO: IMPLEMENT PUNCTUATION SPLITTING
        Steps:
        1. Try splitting on question marks first
        2. If that doesn't work, try periods and exclamation marks
        3. Filter out empty or very short parts
        """
        # TODO: Use self.question_split_pattern.split(query)
        # TODO: Process parts, adding back question marks where appropriate
        # TODO: If question mark splitting doesn't work, try sentence_split_pattern
        # TODO: Return valid parts only if multiple substantial parts found
        return [query]  # Placeholder - replace with your implementation

    def _split_by_conjunctions(self, query: str) -> List[str]:
        """
        Split query by conjunctions and delimiters
        
        TODO: IMPLEMENT CONJUNCTION SPLITTING
        Steps:
        1. Define list of conjunctions to split on
        2. Iteratively split query by each conjunction
        3. Filter and clean resulting parts
        """
        conjunctions = [' & ', '; ', ', and ', ' or ', ' plus ']
        
        # TODO: Start with parts = [query]
        # TODO: For each conjunction, split all existing parts
        # TODO: Filter out empty or very short parts (length <= 3)
        # TODO: Return clean_parts if len > 1, otherwise return [query]
        return [query]  # Placeholder - replace with your implementation

    def clear_cache(self) -> None:
        """Clear the nested query detection cache"""
        self._is_nested_query_cached.cache_clear()
        self._nested_query_cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        return {
            'nested_query_cache_info': self._is_nested_query_cached.cache_info()._asdict(),
            'nested_query_cache_size': len(self._nested_query_cache)
        }