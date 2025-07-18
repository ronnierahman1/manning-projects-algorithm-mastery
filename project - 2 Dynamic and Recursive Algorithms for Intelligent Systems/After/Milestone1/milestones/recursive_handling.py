"""
Optimized Recursive Query Handling

This module implements efficient logic for identifying and processing nested or compound queries.
It breaks down complex questions into simpler sub-questions and processes them individually
to provide informative and structured responses to the user.

Key Concepts:
- Recursive queries: Questions that contain multiple parts or sub-questions
- Query splitting: Breaking complex queries into manageable pieces
- Caching: Storing results to improve performance for repeated queries
- Pattern matching: Using regex to identify different types of compound queries

Example of a recursive query:
"What is Python and how do I install it?" -> ["What is Python?", "How do I install it?"]
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
    
    This class is the main engine for handling complex queries. It uses several strategies:
    1. Pattern recognition to detect compound queries
    2. Multiple splitting methods to break queries apart
    3. Caching to improve performance for repeated queries
    4. Recursion depth limits to prevent infinite loops
    
    Key improvements:
    - Pre-compiled regex patterns for performance
    - Streamlined query detection and splitting
    - Better error handling
    - Reduced code duplication
    - Enhanced maintainability
    """

    def __init__(self, chatbot, max_recursion_depth: int = 5):
        """
        Initialize the RecursiveHandling class.

        Args:
            chatbot: A reference to the main chatbot instance that handles individual queries
            max_recursion_depth: Maximum recursion depth to prevent infinite loops
                                (prevents cases where splitting creates more complex queries)
        """
        # Store reference to the main chatbot for processing individual queries
        self.chatbot = chatbot
        # Set limit to prevent infinite recursion (safety mechanism)
        self.max_recursion_depth = max_recursion_depth
        
        # Pre-compile regex patterns for better performance
        # (Compiling patterns once is faster than compiling them every time we use them)
        self._compile_patterns()
        
        # Cache for nested query detection to avoid re-analyzing the same queries
        self._nested_query_cache = {}

    def _compile_patterns(self) -> None:
        """
        Pre-compile all regex patterns for better performance
        
        Regex compilation is expensive, so we do it once during initialization
        rather than every time we need to match a pattern.
        """
        # Compound query patterns - these detect queries with multiple parts connected by "and"
        # Each pattern captures two groups: the parts before and after the connecting word
        compound_patterns = [
            r'tell me about (.*?) and (.*?)(?:\?|$)',  # "tell me about X and Y"
            r'what about (.*?) and (.*?)(?:\?|$)',     # "what about X and Y"
            r'explain (.*?) and (.*?)(?:\?|$)',        # "explain X and Y"
            r'describe (.*?) and (.*?)(?:\?|$)',       # "describe X and Y"
            r'(.*?) and also (.*?)(?:\?|$)',           # "X and also Y"
            r'(.*?) as well as (.*?)(?:\?|$)'          # "X as well as Y"
        ]
        # Compile all patterns with IGNORECASE flag so they match regardless of capitalization
        self.compound_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in compound_patterns]
        
        # Question boundary pattern - detects when a new question starts after "and"
        # Example: "What is Python and how do I install it?" -> splits at "and how"
        question_words = ['what', 'how', 'why', 'where', 'when', 'who', 'which']
        self.question_boundary_pattern = re.compile(
            r'\b(?:and|&)\s+(' + '|'.join(question_words) + r')\b',  # Match "and/& + question_word"
            re.IGNORECASE
        )
        
        # Punctuation splitting patterns for different types of sentence boundaries
        self.question_split_pattern = re.compile(r'\?')      # Split on question marks
        self.sentence_split_pattern = re.compile(r'[.!]+')   # Split on periods and exclamation marks
        self.clause_split_pattern = re.compile(r'[.?!]+')    # Split on any sentence-ending punctuation

    @lru_cache(maxsize=100)  # Cache the last 100 query checks to improve performance
    def _is_nested_query_cached(self, query: str) -> bool:
        """
        Cached version of nested query detection for frequently asked queries
        
        The @lru_cache decorator automatically caches results, so if we see the same
        query again, we don't need to re-analyze it. This significantly improves
        performance for repeated or similar queries.
        
        Args:
            query: The user input string to analyze
            
        Returns:
            bool: True if the query contains multiple parts/questions
        """
        return self._is_nested_query_internal(query)

    def _is_nested_query_internal(self, query: str) -> bool:
        """
        Internal method to determine if a query contains multiple sub-questions.
        
        This method uses several heuristics (rules of thumb) to detect compound queries:
        1. Look for explicit connecting words like "and", "also", etc.
        2. Count question marks (multiple ? usually means multiple questions)
        3. Look for multiple question words in different sentence parts
        
        Args:
            query: The user input string
            
        Returns:
            bool: True if the query is compound/multi-part
        """
        # Quick validation: very short queries are unlikely to be compound
        if not query or len(query.strip()) < 10:  # Too short to be compound
            return False
            
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Quick indicators for compound queries - words/phrases that often connect multiple questions
        nested_indicators = {
            ' and ', ' & ', '; ', ' or ', ' plus ',        # Direct connectors
            'also', 'additionally', 'furthermore', 'moreover',  # Additive words
            'what about', 'how about', 'tell me about',          # Question starters
            'explain both', 'describe both', 'compare'           # Explicit multi-part requests
        }
        
        # Heuristic 1: Check for explicit indicators of compound queries
        if any(indicator in query_lower for indicator in nested_indicators):
            return True

        # Heuristic 2: Multiple question marks usually indicate multiple questions
        # Example: "What is Python? How do I install it?"
        if query.count('?') > 1:
            return True

        # Heuristic 3: Look for multiple question words in different sentence parts
        # This catches cases like "What is Python and how does it work?"
        question_words = {'what', 'how', 'why', 'where', 'when', 'who', 'which'}
        # Split the query into parts using punctuation
        parts = self.clause_split_pattern.split(query_lower)
        question_parts = []
        
        # Check each part to see if it contains question words
        for part in parts:
            part_words = set(part.strip().split())  # Convert to set for efficient intersection
            if question_words & part_words:  # Intersection check - any question words in this part?
                question_parts.append(part)
        
        # If we found question words in multiple parts, it's likely a compound query
        return len(question_parts) > 1

    def handle_recursive_query(self, query: str, depth: int = 0) -> QueryResult:
        """
        Primary entry point for handling recursive queries.
        
        This is the main method that decides whether to split a query or process it as-is.
        It uses recursion to handle nested queries, but with safety limits to prevent
        infinite loops.
        
        Args:
            query: The user input string (potentially complex)
            depth: Tracks the recursion level to prevent overflow (starts at 0)
            
        Returns:
            QueryResult: Structured result containing response and metadata
        """
        # Record start time to measure processing performance
        start_time = time.time()
        
        try:
            # Guard clause: Prevent excessive recursion that could crash the system
            # If we've gone too deep, just give up and ask the user to simplify
            if depth >= self.max_recursion_depth:
                return QueryResult(
                    response="The query is too complex. Please try breaking it into simpler parts.",
                    responses=[],  # No sub-responses since we couldn't process it
                    processing_time=time.time() - start_time,
                    is_recursive=True,  # Mark as recursive even though we failed
                    used_cache=False,   # We didn't use cache for this error case
                    priority=1,         # Low priority since it's an error
                    fuzzy_match=False,  # No fuzzy matching for error responses
                    fuzzy_match_threshold=0.0
                )

            # Only check for nested queries at the initial call (depth == 0)
            # This prevents us from re-analyzing queries that we've already split
            if depth == 0 and self._is_nested_query_cached(query.strip()):
                # Query is complex - split it and process parts recursively
                return self._handle_nested_query(query, depth + 1, start_time)

            # Process as single query - either it's not complex or we're already in recursion
            return self._handle_single_query(query, start_time)

        except Exception as e:
            # Catch any unexpected errors and provide a graceful fallback
            print(f"Error in handle_recursive_query: {e}")  # Log for debugging
            return QueryResult(
                response="An error occurred while processing your query. Please try rephrasing it.",
                responses=[],
                processing_time=time.time() - start_time,
                is_recursive=True,   # Mark as recursive since we were trying to handle complexity
                used_cache=False,
                priority=1,
                fuzzy_match=False,
                fuzzy_match_threshold=0.0
            )

    def _handle_single_query(self, query: str, start_time: float) -> QueryResult:
        """
        Handle a single, non-nested query
        
        This method processes queries that don't need to be split. It delegates
        the actual query processing to the main chatbot and wraps the result
        in our standard QueryResult format.
        
        Args:
            query: A simple, single-part query
            start_time: When we started processing (for timing)
            
        Returns:
            QueryResult: Standardized result object
        """
        # Delegate to the main chatbot for the actual query processing
        # The chatbot returns: (response_text, is_fuzzy_match, confidence_threshold)
        response_text, is_fuzzy, threshold = self.chatbot.handle_query(query)
        
        # Wrap the result in our standard format
        return QueryResult(
            response=response_text,
            responses=[(response_text, is_fuzzy, threshold)],  # Single response in list format
            processing_time=time.time() - start_time,
            is_recursive=False,  # Not recursive since it's a single query
            used_cache=False,    # Cache usage is handled by the chatbot, not here
            priority=1,          # Single queries get priority 1
            fuzzy_match=is_fuzzy,
            fuzzy_match_threshold=threshold
        )

    def _handle_nested_query(self, query: str, depth: int, start_time: float) -> QueryResult:
        """
        Handle compound queries by splitting and processing each part.
        
        This method is the heart of the recursive processing. It:
        1. Splits the complex query into simpler parts
        2. Processes each part individually (using recursion)
        3. Combines all the results into a coherent response
        
        Args:
            query: A nested user query (contains multiple parts)
            depth: Current recursion depth (for safety)
            start_time: Time when processing started
            
        Returns:
            QueryResult: Combined result from all sub-queries
        """
        try:
            # Step 1: Split the complex query into simpler sub-queries
            sub_queries = self._split_into_subqueries(query)
            
            # If splitting didn't work (returned only 1 part), treat as single query
            if len(sub_queries) <= 1:
                # Fallback to single query if splitting didn't work
                return self._handle_single_query(query, start_time)
            
            # Storage for individual responses and formatted output
            responses = []           # Raw responses for metadata
            formatted_responses = [] # Formatted responses for display
            
            # Step 2: Process each sub-query individually
            for i, sub_query in enumerate(sub_queries):
                sub_query = sub_query.strip()  # Remove extra whitespace
                if not sub_query:  # Skip empty sub-queries
                    continue
                    
                # Process each sub-query recursively (depth > 0 prevents re-nesting)
                # This recursive call handles the possibility that a sub-query might
                # itself be complex, while preventing infinite recursion
                result = self.handle_recursive_query(sub_query, depth + 1)
                sub_resp = result.response
                fuzzy = result.fuzzy_match
                threshold = result.fuzzy_match_threshold
                
                # Store the raw response data for metadata
                responses.append((sub_resp, fuzzy, threshold))
                
                # Format the response for display with clear question/answer structure
                formatted_responses.append(
                    f"**Question {i+1}**: {sub_query}\n**Answer**: {sub_resp}"
                )

            # If no valid responses were generated, fallback to single query processing
            if not responses:
                return self._handle_single_query(query, start_time)

            # Step 3: Combine all responses into a coherent answer
            # Create a structured response that clearly shows each part
            combined_response = "I'll address each part of your question:\n\n" + "\n\n".join(formatted_responses)

            # Step 4: Create metadata for the combined result
            return QueryResult(
                response=combined_response,
                responses=responses,  # All individual responses for reference
                processing_time=time.time() - start_time,
                is_recursive=True,   # Mark as recursive since we split the query
                used_cache=False,    # Cache usage tracked at individual query level
                priority=2,          # Complex queries get higher priority number
                # Aggregate fuzzy matching info: True if ANY sub-query used fuzzy matching
                fuzzy_match=any(r[1] for r in responses),
                # Use the highest confidence threshold from all sub-queries
                fuzzy_match_threshold=max((r[2] for r in responses), default=0.0)
            )

        except Exception as e:
            # Handle any errors gracefully
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
        
        This method tries multiple strategies in order of reliability:
        1. Compound patterns (most reliable) - uses pre-defined regex patterns
        2. Question boundary detection - looks for "and how", "and what", etc.
        3. Punctuation-based splitting - splits on ?, !, .
        4. Conjunction-based splitting - splits on "and", "or", etc.
        
        Args:
            query: The full query text
            
        Returns:
            List[str]: A list of simpler, self-contained queries
        """
        # Quick validation - very short queries don't need splitting
        if not query or len(query.strip()) < 10:
            return [query]
        
        try:
            # Method 1: Try compound patterns first (most reliable)
            # These patterns are specifically designed to catch common compound query structures
            for pattern in self.compound_patterns:
                match = pattern.search(query)
                if match:
                    # Extract the captured groups (the parts before and after connecting words)
                    groups = [group.strip() for group in match.groups() if group and group.strip()]
                    if len(groups) > 1:  # Successfully found multiple parts
                        return groups

            # Method 2: Question boundary detection
            # Looks for patterns like "and how", "and what" that indicate new questions
            parts = self._split_by_question_boundaries(query)
            if len(parts) > 1:
                return parts

            # Method 3: Punctuation-based splitting
            # Splits on question marks, periods, exclamation marks
            parts = self._split_by_punctuation(query)
            if len(parts) > 1:
                return parts

            # Method 4: Conjunction-based splitting
            # Last resort - split on connecting words like "and", "or"
            parts = self._split_by_conjunctions(query)
            if len(parts) > 1:
                return parts

            # If none of the splitting methods worked, return the original query
            return [query]

        except Exception as e:
            # If any error occurs during splitting, just return the original query
            print(f"Error in _split_into_subqueries: {e}")
            return [query]

    def _split_by_question_boundaries(self, query: str) -> List[str]:
        """
        Split query by question word boundaries like 'and where', 'and what'
        
        This method looks for patterns where "and" or "&" is followed by question words,
        which usually indicates the start of a new question.
        
        Example: "What is Python and how do I install it?"
        -> Splits at "and how" -> ["What is Python", "how do I install it?"]
        """
        # Find all matches of our question boundary pattern
        matches = list(self.question_boundary_pattern.finditer(query))
        
        # If no boundaries found, return original query
        if not matches:
            return [query]
        
        parts = []
        start = 0  # Track where each part begins
        
        # Process each boundary we found
        for match in matches:
            # Add the part before the question word
            part_before = query[start:match.start()].strip()
            if part_before and len(part_before) > 3:  # Only include substantial parts
                parts.append(part_before)
            
            # Start next part from the question word (captured in group 1)
            start = match.start(1)  # Start from the question word, not the "and"
        
        # Add the remaining part after the last boundary
        remaining = query[start:].strip()
        if remaining and len(remaining) > 3:
            parts.append(remaining)
        
        # Only return split result if we got multiple valid parts
        return parts if len(parts) > 1 else [query]

    def _split_by_punctuation(self, query: str) -> List[str]:
        """
        Split query by punctuation marks
        
        This method tries different punctuation marks to split the query:
        1. Question marks (?) - most reliable for questions
        2. Periods and exclamation marks (., !) - for statements
        """
        # Try question marks first (most reliable for splitting questions)
        question_parts = self.question_split_pattern.split(query)
        if len(question_parts) > 1:
            valid_parts = []
            # Process each part from the split
            for i, part in enumerate(question_parts):
                part = part.strip()
                if part and len(part) > 3:  # Only include substantial parts
                    # Add back the question mark except for the very last part (if it's empty)
                    if i < len(question_parts) - 1 or not question_parts[-1].strip():
                        part += '?'
                    valid_parts.append(part)
            
            # Return if we got multiple valid parts
            if len(valid_parts) > 1:
                return valid_parts
        
        # Try other punctuation if question marks didn't work
        sentences = self.sentence_split_pattern.split(query)
        if len(sentences) > 1:
            # Filter out empty or very short parts
            valid_parts = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
            if len(valid_parts) > 1:
                return valid_parts
        
        # If no punctuation splitting worked, return original
        return [query]

    def _split_by_conjunctions(self, query: str) -> List[str]:
        """
        Split query by conjunctions and delimiters
        
        This is the most aggressive splitting method - it splits on common
        connecting words. This is a last resort because it can sometimes
        split single concepts incorrectly.
        
        Example: "cats and dogs" shouldn't be split, but "what are cats and what are dogs" should be.
        """
        # List of conjunctions/delimiters to split on
        conjunctions = [' & ', '; ', ', and ', ' or ', ' plus ']
        parts = [query]  # Start with the whole query
        
        # Apply each conjunction splitting iteratively
        for conj in conjunctions:
            new_parts = []
            # Split each existing part by the current conjunction
            for part in parts:
                if conj in part:
                    # Split this part and add all pieces
                    new_parts.extend(part.split(conj))
                else:
                    # No conjunction found, keep the part as-is
                    new_parts.append(part)
            parts = new_parts
        
        # Filter and clean parts - remove empty or very short parts
        clean_parts = [part.strip() for part in parts if part.strip() and len(part.strip()) > 3]
        
        # Only return split result if we got multiple valid parts
        return clean_parts if len(clean_parts) > 1 else [query]

    def clear_cache(self) -> None:
        """
        Clear the nested query detection cache
        
        This method clears all cached data. Useful for:
        - Freeing memory
        - Forcing re-analysis of queries (if the detection logic changed)
        - Testing purposes
        """
        # Clear the LRU cache used by _is_nested_query_cached
        self._is_nested_query_cached.cache_clear()
        # Clear our manual cache dictionary
        self._nested_query_cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring
        
        This method provides insights into cache performance, which is useful for:
        - Performance monitoring
        - Debugging cache-related issues
        - Optimizing cache size
        
        Returns:
            Dict containing cache statistics
        """
        return {
            # Get statistics from the LRU cache (hits, misses, size, etc.)
            'nested_query_cache_info': self._is_nested_query_cached.cache_info()._asdict(),
            # Get size of our manual cache
            'nested_query_cache_size': len(self._nested_query_cache)
        }