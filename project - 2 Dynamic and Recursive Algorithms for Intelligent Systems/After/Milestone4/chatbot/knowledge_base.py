"""
Knowledge Base Module - Comprehensive Educational Documentation

This module manages the chatbot's QA knowledge base with advanced features:
- Loads SQuAD-style JSON data from file with fallback mechanisms
- Supports exact and fuzzy search with unified fuzzy matching logic
- Provides comprehensive analytics, caching, and performance optimization
- Implements persistent storage for custom data and user feedback

Educational Focus Areas:
- File I/O and JSON processing with error handling
- Text normalization and preprocessing techniques
- Fuzzy string matching algorithms using difflib
- Caching strategies for performance optimization
- Data persistence and serialization patterns
- Analytics and metrics collection
- Object-oriented design patterns

Milestone: Implemented fully in Milestone 4.
Enhanced: Added advanced search, analytics, caching, and persistence features.
Refactored: Unified fuzzy matching and text processing utilities.
"""

import json          # For parsing and writing JSON data files
import os            # For file system operations (checking file existence)
import difflib       # For fuzzy string matching using sequence comparison algorithms
import string        # For string constants like punctuation characters
import re            # For regular expression pattern matching
import time          # For performance timing measurements
from typing import List, Dict, Any, Optional, Tuple  # Type hints for better code documentation
from collections import defaultdict, Counter          # Specialized dictionary types for counting/grouping
from datetime import datetime                         # For timestamping and date operations
import pickle        # For binary serialization (though not actively used in this implementation)
import hashlib       # For creating hash digests for caching keys


class KnowledgeBase:
    """
    Advanced Knowledge Base class that manages QA data with enterprise-level features.
    
    This class demonstrates several important software engineering patterns:
    
    **Factory Pattern**: Creates appropriate data structures based on input
    **Strategy Pattern**: Multiple search strategies (exact, fuzzy, keyword)
    **Observer Pattern**: Analytics tracking observes search operations
    **Cache Pattern**: Multiple levels of caching for performance
    **Singleton-like**: Manages shared state for analytics and caching
    
    Key Educational Concepts:
    - Text processing and normalization techniques
    - Fuzzy matching algorithms and threshold tuning
    - Performance optimization through caching
    - Data persistence and serialization
    - Analytics collection and reporting
    - Error handling and graceful degradation
    
    Architecture:
    - Core QA storage and retrieval
    - Multiple search strategies with fallbacks
    - Performance monitoring and optimization
    - User feedback collection and analysis
    - Custom data management with persistence
    """

    # Class-level constants: These are shared across all instances and define system behavior
    # Using decreasing thresholds allows for graceful degradation in search quality
    # Higher thresholds (0.9) require very close matches, lower ones (0.5) are more permissive
    DEFAULT_FUZZY_THRESHOLDS = [0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
    
    # Pre-compiled regex pattern for removing sentence-ending punctuation
    # Compiling once at class level is more efficient than compiling repeatedly
    # This pattern matches one or more punctuation marks at the end of strings
    PUNCTUATION_PATTERN = re.compile(r'[.?!]+$')

    def __init__(self, data_path: str):
        """
        Initialize the knowledge base with comprehensive feature set.

        Args:
            data_path (str): Path to the dev-v2.0.json file (SQuAD format)
            
        The initialization follows a multi-phase approach:
        1. Core data structures setup
        2. Data loading with fallback mechanisms
        3. Search index building for performance
        4. Enhancement features loading from persistent storage
        """
        # Core data storage: Main repository of question-answer pairs loaded from JSON
        self.qa_pairs = []        # Cached list of all parsed QA entries from main dataset
        
        # Store the data path for potential reloading or reference
        self.data_path = data_path
        
        # === Enhanced Analytics and Performance Features ===
        # These features transform a basic QA system into an enterprise-ready solution
        
        # Analytics: Track how the system is being used for optimization insights
        # defaultdict automatically creates missing keys with default values (int() = 0)
        self.search_analytics = defaultdict(int)  # Track search patterns and success rates
        
        # Performance Optimization: Multiple caching layers
        self.query_cache = {}     # Cache for frequently asked questions (query -> answer)
        
        # Content Enhancement: Improve search capabilities
        self.synonyms = {}        # Synonym dictionary for better matching (future enhancement)
        self.categories = defaultdict(list)  # Categorized QA pairs for topic-based search
        
        # Monitoring and Debugging: Track system behavior over time
        self.search_history = []  # Track search history with timestamps and performance
        self.response_times = []  # Track performance metrics for optimization
        
        # User Experience: Collect and analyze user satisfaction
        self.feedback_data = {}   # Store user feedback on answers for quality improvement
        
        # Extensibility: Allow runtime addition of new knowledge
        self.custom_qa_pairs = [] # User-added QA pairs that persist between sessions
        
        # Performance: Pre-computed search indices for faster lookups
        self.index_cache = {}     # Cached search indices (questions, answers, contexts)
        
        # === Initialization Sequence ===
        # The order of these operations is important for proper system startup
        
        # Phase 1: Load core data from JSON file with fallback to sample data
        self.load_data(data_path)
        
        # Phase 2: Build search indices for performance optimization
        self._build_search_index()
        
        # Phase 3: Load enhancement data from persistent storage
        self._load_enhancements()

    # ============================================================================
    # Text Processing Utilities - Foundation of all text-based operations
    # ============================================================================

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for consistent comparison across the system.
        
        This static method implements a standardized text preprocessing pipeline
        that ensures all text comparisons use the same normalization rules.
        
        Normalization Steps:
        1. Strip leading/trailing whitespace
        2. Convert to lowercase for case-insensitive matching
        3. Remove sentence-ending punctuation (., !, ?)
        4. Strip again to remove any trailing spaces after punctuation removal
        
        Args:
            text (str): Raw text to normalize
            
        Returns:
            str: Normalized text ready for comparison
            
        Static method is used because this utility doesn't need instance data
        and can be reused independently. This promotes code reusability and
        makes the normalization logic easily testable.
        """
        # Handle edge case: empty or None input
        if not text:
            return ""
            
        # Pipeline approach: each step transforms the text further
        # 1. Strip whitespace and convert to lowercase
        normalized = text.strip().lower()
        
        # 2. Remove ending punctuation using pre-compiled regex
        # sub() replaces matches with empty string, effectively removing them
        normalized = KnowledgeBase.PUNCTUATION_PATTERN.sub('', normalized)
        
        # 3. Final cleanup: remove any trailing spaces left after punctuation removal
        return normalized.strip()

    @staticmethod
    def clean_words(text: str) -> set:
        """
        Extract and clean individual words from text for word-based matching.
        
        This method is crucial for query processing that relies on word intersection
        rather than full string matching. It handles punctuation removal and
        ensures consistent word extraction across the system.
        
        Processing Pipeline:
        1. Convert to lowercase for consistency
        2. Split into individual words
        3. Remove all punctuation from each word
        4. Filter out empty strings
        5. Return as set for O(1) lookup performance
        
        Args:
            text (str): Text to process into clean words
            
        Returns:
            set: Set of cleaned words (duplicates automatically removed)
            
        Using a set provides several advantages:
        - O(1) lookup time for membership testing
        - Automatic duplicate removal
        - Set operations (intersection, union) for advanced matching
        """
        # Handle edge case: empty or None input
        if not text:
            return set()
            
        # Step 1: Convert to lowercase and split by whitespace
        # split() without arguments handles multiple spaces, tabs, etc.
        words = text.lower().split()
        
        # Step 2: Clean each word individually
        cleaned_words = set()
        for word in words:
            # Remove all punctuation using generator expression with string.punctuation
            # string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
            clean_word = ''.join(char for char in word if char not in string.punctuation)
            
            # Only add non-empty words to avoid meaningless entries
            if clean_word:  # Truthy check: excludes empty strings
                cleaned_words.add(clean_word)
        
        return cleaned_words

    # ============================================================================
    # Unified Fuzzy Matching - Core search algorithm implementation
    # ============================================================================

    def fuzzy_match(self, query: str, threshold: float = 0.6, max_results: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform fuzzy string matching against the entire knowledge base.
        
        This method implements the core search algorithm using Python's difflib
        for sequence-based similarity matching. It demonstrates important concepts:
        
        **Algorithm**: Uses SequenceMatcher which implements Ratcliff-Obershelp algorithm
        **Performance**: O(n*m) where n=query length, m=database size
        **Accuracy**: Provides similarity scores from 0.0 (no match) to 1.0 (exact match)
        
        Args:
            query (str): User's search query
            threshold (float): Minimum similarity threshold (0.0 to 1.0)
            max_results (int): Maximum number of results to return
            
        Returns:
            List of tuples containing (qa_dictionary, similarity_score) sorted by score
            
        The method implements defensive programming with comprehensive error handling
        and graceful degradation when problems occur.
        """
        try:
            # Input validation: Early return for invalid inputs
            if not query or not query.strip():
                return []
                
            # Normalize query using standardized preprocessing
            query_normalized = self.normalize_text(query)
            if not query_normalized:
                return []
                
            # Initialize results collection
            matches = []
            
            # Combine all available QA pairs: main dataset + custom additions
            # getattr() with default provides safe access to attributes that might not exist
            all_qa_pairs = getattr(self, 'qa_pairs', []) + getattr(self, 'custom_qa_pairs', [])
            
            # Core matching loop: Compare query against every QA pair
            for qa in all_qa_pairs:
                # Defensive programming: Validate QA structure before processing
                if not qa or 'question' not in qa:
                    continue
                    
                # Normalize the stored question for fair comparison
                question_normalized = self.normalize_text(qa['question'])
                
                # Calculate similarity using difflib's SequenceMatcher
                # SequenceMatcher(None, a, b) compares sequences a and b
                # ratio() returns similarity as float between 0.0 and 1.0
                import difflib
                similarity = difflib.SequenceMatcher(None, query_normalized, question_normalized).ratio()
                
                # Threshold filtering: Only keep matches above minimum similarity
                if similarity >= threshold:
                    matches.append((qa, similarity))
            
            # Sort results by similarity score in descending order (best matches first)
            # lambda function extracts similarity score (second element of tuple) for sorting
            matches.sort(key=lambda x: x[1], reverse=True)
            
            # Limit results to requested maximum to control response size
            return matches[:max_results]
            
        except Exception as e:
            # Error handling: Log error but don't crash the system
            print(f"Error in fuzzy_match: {e}")
            return []  # Always return empty list on error to maintain consistent interface
        
    def fuzzy_match_with_thresholds(self, query: str, thresholds: List[float] = None) -> Optional[Tuple[str, float]]:
        """
        Advanced fuzzy matching with cascading thresholds for optimal results.
        
        This method implements a sophisticated search strategy that tries multiple
        similarity thresholds in descending order. This approach balances accuracy
        with recall:
        
        **High Thresholds (0.9)**: Very precise matches, low recall
        **Medium Thresholds (0.7-0.8)**: Balanced precision and recall  
        **Low Thresholds (0.5-0.6)**: High recall, lower precision
        
        The cascading approach ensures we get the most precise match available
        while still providing results if no high-precision matches exist.
        
        Args:
            query (str): User's search query
            thresholds (List[float]): Ordered list of thresholds to try (highest to lowest)
            
        Returns:
            Tuple of (answer_text, similarity_score) or None if no match found
            
        This method is critical for the chatbot's query processing pipeline
        and demonstrates advanced search algorithm design.
        """
        # Use default thresholds if none provided
        # This provides a sensible default while allowing customization
        if thresholds is None:
            thresholds = self.DEFAULT_FUZZY_THRESHOLDS
            
        try:
            # Cascade through thresholds from highest (most strict) to lowest (most permissive)
            for threshold in thresholds:
                # Try fuzzy matching at current threshold level
                # max_results=1 because we want the single best match at this threshold
                matches = self.fuzzy_match(query, threshold, max_results=1)
                
                # Check if we found any matches at this threshold
                if matches and len(matches) > 0:
                    # Extract the best match (guaranteed to be first due to sorting)
                    qa_dict, similarity = matches[0]
                    
                    # Update usage statistics for analytics and optimization
                    self._update_usage_stats(qa_dict)
                    
                    # Extract answer text from QA dictionary
                    answer = qa_dict.get('answer', '')
                    
                    # Return tuple with exactly 2 elements as expected by caller
                    # This explicit tuple construction ensures consistent return format
                    return (answer, similarity)
                    
            # If no thresholds yielded results, explicitly return None
            # This is clearer than falling through and returning nothing
            return None
            
        except Exception as e:
            # Comprehensive error handling: Log but don't crash
            print(f"Error in fuzzy_match_with_thresholds: {e}")
            # Critical: Always return None on error, never an empty tuple or other type
            # This maintains the Optional[Tuple[str, float]] return type contract
            return None
    
    # ============================================================================
    # Main Query Processing - Core business logic
    # ============================================================================

    def load_data(self, data_path: str) -> None:
        """
        Load and parse SQuAD-style JSON dataset into memory with robust error handling.

        This method implements a comprehensive data loading strategy that handles:
        - File existence validation
        - JSON format validation
        - Data structure validation
        - Graceful fallback to sample data
        - Performance optimization through caching

        SQuAD Format Structure:
        {
          "data": [
            {
              "paragraphs": [
                {
                  "context": "paragraph text...",
                  "qas": [
                    {
                      "question": "question text?",
                      "answers": [{"text": "answer text"}]
                    }
                  ]
                }
              ]
            }
          ]
        }

        Args:
            data_path (str): Path to the JSON file containing QA data
            
        The method follows the fail-safe principle: if anything goes wrong,
        it creates usable sample data rather than leaving the system unusable.
        """
        # Step 1: File existence validation
        # Check if file exists before attempting to read it
        if not os.path.exists(data_path):
            print(f"⚠️ Warning: {data_path} not found.")
            self._create_sample_data()  # Fallback to sample data
            return

        try:
            # Step 2: File reading with proper encoding
            # UTF-8 encoding ensures international characters are handled correctly
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)  # Parse JSON into Python dictionary

            # Step 3: Data structure validation
            # Verify the JSON has the expected SQuAD format structure
            if "data" not in data:
                raise ValueError("JSON format is invalid. Missing 'data' key.")

            # Step 4: Nested data parsing
            # Navigate through the nested SQuAD structure to extract QA pairs
            for article in data["data"]:                          # Each article contains paragraphs
                for paragraph in article.get("paragraphs", []):   # Each paragraph contains context + QAs
                    context = paragraph.get("context", "")        # Background text for questions
                    
                    for qa in paragraph.get("qas", []):           # Each QA contains question + answers
                        question = qa.get("question", "").strip() # Extract and clean question
                        
                        # Skip invalid questions
                        if not question:
                            continue

                        # Step 5: Answer extraction with fallback logic
                        answers = qa.get("answers", [])
                        if answers:
                            # Use first answer if multiple answers exist
                            answer = answers[0].get("text", "").strip()
                        else:
                            # Fallback: Generate answer from context if no explicit answer
                            # Truncate context to reasonable length for readability
                            answer = f"Based on the context: {context[:200]}..." if context else "Answer not available."

                        # Step 6: Create structured QA entry with metadata
                        # Only store valid question-answer pairs
                        if question and answer:
                            qa_entry = {
                                "question": question,
                                "answer": answer,
                                "context": context,
                                "id": len(self.qa_pairs),        # Unique identifier for this QA pair
                                "usage_count": 0,                # Analytics: track how often this is accessed
                                "last_used": None,               # Analytics: when was this last accessed
                                "confidence": 1.0                # Quality score for this QA pair
                            }
                            self.qa_pairs.append(qa_entry)

            # Success notification with statistics
            print(f"✅ Loaded {len(self.qa_pairs)} QA pairs from knowledge base.")

        except Exception as e:
            # Comprehensive error handling: catch any parsing or file reading errors
            print(f"❌ Failed to load knowledge base: {e}")
            self._create_sample_data()  # Always provide fallback data

    def _create_sample_data(self) -> None:
        """
        Create fallback sample data when JSON loading fails.
        
        This method ensures the system remains functional even when the main
        data source is unavailable. It provides a minimal but complete dataset
        that demonstrates the system's capabilities.
        
        The sample data follows the same structure as loaded data, ensuring
        compatibility with all system components.
        """
        # Create minimal but functional QA pairs
        # Each entry follows the same structure as JSON-loaded data
        self.qa_pairs = [
            {
                'question': 'What is AI?',                        # Simple question for testing
                'answer': 'AI stands for artificial intelligence.', # Clear, concise answer
                'context': 'AI basics',                           # Topic categorization
                'id': 0,                                          # Unique identifier
                'usage_count': 0,                                 # Analytics tracking
                'last_used': None,                                # Last access time
                'confidence': 1.0                                 # Quality confidence score
            },
            {
                'question': 'What is the capital of France?',
                'answer': 'The capital of France is Paris.',
                'context': 'World geography',
                'id': 1,
                'usage_count': 0,
                'last_used': None,
                'confidence': 1.0
            }
        ]
        print("✅ Created fallback QA pairs.")

    def size(self) -> int:
        """
        Return total number of available QA pairs including custom additions.
        
        This method provides a simple way to check the knowledge base size
        for analytics and system monitoring purposes.
        
        Returns:
            int: Total count of QA pairs from all sources
        """
        # Combine main dataset size with custom additions
        return len(self.qa_pairs) + len(self.custom_qa_pairs)

    def get_exact_match_answer(self, query: str) -> str:
        """
        Primary query processor implementing exact matching with comprehensive features.
        
        This method serves as the main entry point for query processing and
        demonstrates several advanced software engineering concepts:
        
        **Performance Monitoring**: Tracks response times for optimization
        **Caching Strategy**: Implements query result caching with LRU eviction
        **Analytics Integration**: Records search patterns and success rates
        **Defensive Programming**: Comprehensive input validation and error handling
        
        Processing Pipeline:
        1. Input validation and preprocessing
        2. Cache lookup for performance optimization
        3. Exact string matching against normalized questions
        4. Analytics and usage tracking
        5. Result caching for future queries
        
        Args:
            query (str): User's question to find an answer for
            
        Returns:
            str: Answer text or appropriate fallback message
            
        The method prioritizes exact matches for accuracy while providing
        fallback messages that guide users toward successful queries.
        """
        # Performance monitoring: Start timing this query
        start_time = time.time()
        
        try:
            # Step 1: Input validation with early return pattern
            if not query or not query.strip():
                return "Please enter a question."

            # Step 2: Cache lookup for performance optimization
            # Hash the query to create a unique cache key
            query_hash = self._hash_query(query)
            if query_hash in self.query_cache:
                # Cache hit: return cached result and update analytics
                self._update_analytics(query, True)
                return self.query_cache[query_hash]

            # Step 3: Normalize query for consistent comparison
            query_normalized = self.normalize_text(query)

            # Step 4: Exact matching against all available QA pairs
            # Combine main dataset with custom additions for comprehensive search
            all_qa_pairs = getattr(self, 'qa_pairs', []) + getattr(self, 'custom_qa_pairs', [])
            
            for qa in all_qa_pairs:
                # Defensive programming: validate QA structure
                if not qa or 'question' not in qa:
                    continue
                    
                # Normalize stored question for fair comparison
                question_normalized = self.normalize_text(qa['question'])
                
                # Check for exact match using normalized strings
                if query_normalized == question_normalized:
                    answer = qa.get('answer', '')
                    
                    # Update analytics and usage tracking
                    self._update_usage_stats(qa)      # Track usage for this QA pair
                    self._cache_result(query_hash, answer)  # Cache for future queries
                    self._update_analytics(query, True)     # Record successful search
                    
                    return answer

            # No exact match found: provide helpful fallback message
            return f"I couldn't find a good match for: '{query}'. Please rephrase."

        except Exception as e:
            # Comprehensive error handling: log error but provide usable response
            print(f"❌ Error in get_exact_match_answer: {e}")
            return "An error occurred while processing your question."
        
        finally:
            # Performance tracking: Always execute regardless of success/failure
            # This ensures we collect timing data for system optimization
            response_time = time.time() - start_time
            
            # Store performance metrics if analytics are enabled
            if hasattr(self, 'response_times'):
                self.response_times.append(response_time)
                
            # Store search history if tracking is enabled
            if hasattr(self, 'search_history'):
                self.search_history.append({
                    'query': query,
                    'timestamp': datetime.now(),
                    'response_time': response_time
                })

    # ============================================================================
    # Enhanced Search and Analytics Methods
    # ============================================================================

    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search with relevance scoring across all QA content.
        
        This method implements a content-based search that looks beyond exact
        question matching to find relevant information in questions, answers,
        and context. It demonstrates information retrieval concepts:
        
        **Term Frequency**: How often the keyword appears in different fields
        **Field Weighting**: Questions weighted higher than context
        **Relevance Scoring**: Combines multiple signals for ranking
        
        Args:
            keyword (str): Search term to look for across QA content
            limit (int): Maximum number of results to return
            
        Returns:
            List of QA dictionaries with added relevance scores, sorted by relevance
            
        This search method complements exact and fuzzy matching by providing
        exploratory search capabilities for content discovery.
        """
        # Normalize keyword for case-insensitive matching
        keyword_lower = keyword.lower()
        results = []
        
        # Search across all available QA pairs
        for qa in self.qa_pairs + self.custom_qa_pairs:
            score = 0  # Initialize relevance score
            
            # Field-weighted scoring system:
            # Questions are most important (weight: 3)
            if keyword_lower in qa['question'].lower():
                score += 3
            
            # Answers are moderately important (weight: 2)
            if keyword_lower in qa['answer'].lower():
                score += 2
                
            # Context provides background relevance (weight: 1)
            if keyword_lower in qa.get('context', '').lower():
                score += 1
                
            # Only include results with some relevance
            if score > 0:
                # Create result copy to avoid modifying original data
                result = qa.copy()
                result['relevance_score'] = score
                results.append(result)
        
        # Sort by relevance score (highest first) and limit results
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]

    def get_similar_questions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find questions semantically similar to the given query for exploration.
        
        This method helps users discover related information when their exact
        question isn't available. It uses fuzzy matching with a lower threshold
        to cast a wider net for potentially relevant content.
        
        Args:
            query (str): Query to find similar questions for
            limit (int): Maximum number of similar questions to return
            
        Returns:
            List of similar questions with similarity scores and usage data
            
        The method balances similarity with popularity to surface both
        relevant and well-tested content.
        """
        # Use fuzzy matching with lower threshold for broader results
        # threshold=0.3 casts a wide net for potentially related content
        matches = self.fuzzy_match(query, threshold=0.3, max_results=limit)
        
        # Transform matches into user-friendly format
        similarities = []
        for qa, similarity in matches:
            similarities.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'similarity': similarity,
                'usage_count': qa.get('usage_count', 0)  # Include popularity data
            })
        
        return similarities

    def add_custom_qa(self, question: str, answer: str, context: str = "") -> bool:
        """
        Add new knowledge to the system at runtime with persistence.
        
        This method enables the knowledge base to grow and adapt based on
        user needs. It demonstrates dynamic system extension and data
        persistence patterns.
        
        Args:
            question (str): Question text to add
            answer (str): Corresponding answer text
            context (str): Optional context information
            
        Returns:
            bool: True if successfully added, False if validation failed
            
        The method includes validation, unique ID generation, and automatic
        persistence to ensure data survives system restarts.
        """
        try:
            # Input validation: ensure required fields have content
            if not question.strip() or not answer.strip():
                return False
                
            # Create structured QA entry following standard format
            qa_entry = {
                'question': question.strip(),        # Clean whitespace
                'answer': answer.strip(),            # Clean whitespace
                'context': context.strip(),          # Optional context
                'id': len(self.qa_pairs) + len(self.custom_qa_pairs),  # Unique ID
                'usage_count': 0,                    # Initialize analytics
                'last_used': None,                   # No usage yet
                'confidence': 1.0,                   # High confidence for manual entries
                'custom': True                       # Flag as user-added content
            }
            
            # Add to custom QA collection
            self.custom_qa_pairs.append(qa_entry)
            
            # Persist to disk for durability across restarts
            self._save_custom_data()
            return True
            
        except Exception as e:
            # Handle errors gracefully without crashing
            print(f"❌ Error adding custom QA: {e}")
            return False

    def get_analytics(self) -> Dict[str, Any]:
        """
        Generate comprehensive analytics report about knowledge base usage.
        
        This method demonstrates metrics collection and reporting for system
        monitoring and optimization. It provides insights into:
        - System performance and efficiency
        - User behavior and content popularity  
        - Cache effectiveness
        - Success rates and quality metrics
        
        Returns:
            Dictionary containing detailed analytics across multiple dimensions
            
        Analytics help administrators understand system usage patterns
        and identify opportunities for improvement.
        """
        # Calculate search success metrics
        total_queries = sum(self.search_analytics.values())
        successful_queries = self.search_analytics.get('successful', 0)
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        # Identify most popular content for content strategy
        # Sort by usage count to find most valuable QA pairs
        popular_questions = sorted(
            [(qa['question'], qa.get('usage_count', 0)) 
             for qa in self.qa_pairs + self.custom_qa_pairs],
            key=lambda x: x[1], reverse=True  # Sort by usage count descending
        )[:10]  # Top 10 most popular
        
        # Calculate average response time for performance monitoring
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        # Compile comprehensive analytics report
        return {
            'total_qa_pairs': self.size(),
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': f"{success_rate:.1f}%",
            'popular_questions': popular_questions,
            'avg_response_time': f"{avg_response_time:.3f}s",
            'cache_hit_rate': f"{len(self.query_cache) / max(total_queries, 1) * 100:.1f}%",
            'custom_qa_count': len(self.custom_qa_pairs)
        }

    def get_recent_searches(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve recent search history for debugging and user experience analysis.
        
        Args:
            limit (int): Maximum number of recent searches to return
            
        Returns:
            List of recent search entries with timestamps and performance data
            
        This method supports troubleshooting and user experience optimization
        by providing visibility into recent system interactions.
        """
        # Return most recent searches using negative indexing
        # [-limit:] gets the last 'limit' items from the list
        return self.search_history[-limit:]

    def provide_feedback(self, query: str, helpful: bool, comment: str = "") -> None:
        """
        Collect user feedback for continuous quality improvement.
        
        This method implements a feedback loop that enables the system to
        learn from user satisfaction and identify areas for improvement.
        
        Args:
            query (str): The original query that was answered
            helpful (bool): Whether the user found the answer helpful
            comment (str): Optional detailed feedback from the user
            
        Feedback data drives quality improvements and helps prioritize
        content updates and system enhancements.
        """
        # Create unique identifier for the query
        query_hash = self._hash_query(query)
        
        # Store structured feedback data
        self.feedback_data[query_hash] = {
            'helpful': helpful,
            'comment': comment,
            'timestamp': datetime.now()
        }
        
        # Persist feedback for long-term analysis
        self._save_feedback_data()

    def export_knowledge_base(self, filepath: str) -> bool:
        """
        Export complete knowledge base to JSON file for backup or analysis.
        
        This method provides data portability and backup capabilities,
        essential for enterprise deployments. It includes all data sources
        and analytics for comprehensive system state capture.
        
        Args:
            filepath (str): Destination path for the exported JSON file
            
        Returns:
            bool: True if export succeeded, False if an error occurred
            
        The export includes both core data and analytics for complete
        system state preservation.
        """
        try:
            # Compile comprehensive export data
            export_data = {
                'qa_pairs': self.qa_pairs,              # Core dataset
                'custom_qa_pairs': self.custom_qa_pairs, # User additions
                'analytics': self.get_analytics(),       # Usage statistics
                'export_timestamp': datetime.now().isoformat()  # When exported
            }
            
            # Write to file with proper encoding and formatting
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, 
                         indent=2,           # Pretty formatting
                         ensure_ascii=False, # Support international characters
                         default=str)        # Handle datetime objects
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting knowledge base: {e}")
            return False

    def clear_cache(self) -> None:
        """
        Clear the query cache for memory management or testing.
        
        This method provides explicit cache management for scenarios like:
        - Memory optimization in long-running systems
        - Testing with fresh cache state
        - Force cache refresh after system updates
        """
        self.query_cache.clear()
        print("✅ Query cache cleared.")

    def optimize_knowledge_base(self) -> Dict[str, Any]:
        """
        Optimize the knowledge base by removing duplicates and rebuilding indices.
        
        This method implements database-like optimization techniques for
        maintaining system performance over time. It demonstrates:
        - Duplicate detection and removal
        - Index rebuilding for performance
        - Space optimization reporting
        
        Returns:
            Dictionary with optimization statistics and results
            
        Regular optimization maintains system performance as the knowledge
        base grows and evolves.
        """
        original_size = len(self.qa_pairs)
        
        # Duplicate detection using normalized question text
        seen_questions = set()
        unique_qa_pairs = []
        
        for qa in self.qa_pairs:
            # Use normalized question as deduplication key
            q_normalized = self.normalize_text(qa['question'])
            if q_normalized not in seen_questions:
                seen_questions.add(q_normalized)
                unique_qa_pairs.append(qa)
        
        # Calculate optimization statistics
        duplicates_removed = original_size - len(unique_qa_pairs)
        self.qa_pairs = unique_qa_pairs
        
        # Rebuild search indices for optimal performance
        self._build_search_index()
        
        # Return comprehensive optimization report
        return {
            'original_size': original_size,
            'optimized_size': len(self.qa_pairs),
            'duplicates_removed': duplicates_removed,
            'space_saved': f"{duplicates_removed / original_size * 100:.1f}%" if original_size > 0 else "0%"
        }

    # ============================================================================
    # Private Helper Methods - Internal system operations
    # ============================================================================

    def _build_search_index(self) -> None:
        """
        Build search indices for faster content lookups.
        
        This method pre-processes content into searchable indices to improve
        query performance. It demonstrates database indexing concepts applied
        to in-memory data structures.
        
        The indices enable faster keyword and content-based searches by
        avoiding repeated text processing during queries.
        """
        # Create separate indices for different content types
        # Pre-converting to lowercase avoids repeated case conversion during searches
        self.index_cache = {
            'questions': [qa['question'].lower() for qa in self.qa_pairs],
            'answers': [qa['answer'].lower() for qa in self.qa_pairs],
            'contexts': [qa.get('context', '').lower() for qa in self.qa_pairs]
        }

    def _load_enhancements(self) -> None:
        """
        Load enhancement data from persistent storage.
        
        This method restores system state from previous sessions, including
        custom QA pairs and user feedback. It demonstrates data persistence
        and system state restoration patterns.
        
        The method handles missing files gracefully to support fresh installations
        while preserving data when available.
        """
        try:
            # Load custom QA pairs if available
            if os.path.exists('custom_qa.json'):
                with open('custom_qa.json', 'r', encoding='utf-8') as f:
                    self.custom_qa_pairs = json.load(f)
            
            # Load feedback data with datetime reconstruction
            if os.path.exists('feedback.json'):
                with open('feedback.json', 'r', encoding='utf-8') as f:
                    feedback_raw = json.load(f)
                    
                    # Convert string timestamps back to datetime objects
                    # JSON doesn't natively support datetime, so we store as ISO strings
                    for key, value in feedback_raw.items():
                        if 'timestamp' in value:
                            value['timestamp'] = datetime.fromisoformat(value['timestamp'])
                    self.feedback_data = feedback_raw
                    
        except Exception as e:
            # Graceful handling of missing or corrupted enhancement data
            print(f"⚠️ Warning: Could not load enhancement data: {e}")

    def _save_custom_data(self) -> None:
        """
        Persist custom QA pairs to disk for durability.
        
        This method ensures user-added content survives system restarts
        by writing custom QA pairs to a JSON file.
        """
        try:
            with open('custom_qa.json', 'w', encoding='utf-8') as f:
                json.dump(self.custom_qa_pairs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Warning: Could not save custom QA data: {e}")

    def _save_feedback_data(self) -> None:
        """
        Persist feedback data to disk with datetime serialization.
        
        This method handles the complexity of serializing datetime objects
        to JSON format while preserving the data structure.
        """
        try:
            # Convert datetime objects to strings for JSON serialization
            # JSON doesn't support datetime objects natively
            feedback_serializable = {}
            for key, value in self.feedback_data.items():
                feedback_serializable[key] = value.copy()
                if 'timestamp' in value:
                    feedback_serializable[key]['timestamp'] = value['timestamp'].isoformat()
            
            # Write serializable data to file
            with open('feedback.json', 'w', encoding='utf-8') as f:
                json.dump(feedback_serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Warning: Could not save feedback data: {e}")

    def _hash_query(self, query: str) -> str:
        """
        Create a consistent hash key for query caching.
        
        This method generates unique identifiers for queries to use as
        cache keys. It demonstrates cryptographic hashing for creating
        consistent, collision-resistant identifiers.
        
        Args:
            query (str): Query text to hash
            
        Returns:
            str: MD5 hash of normalized query text
            
        The hash provides a fixed-length key regardless of query length
        and ensures identical queries always produce the same key.
        """
        # Normalize query and encode to bytes for hashing
        # MD5 provides fast hashing with low collision probability for cache keys
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def _cache_result(self, query_hash: str, answer: str) -> None:
        """
        Cache a query result with LRU-style eviction policy.
        
        This method implements a simple cache management strategy that
        prevents unlimited memory growth while preserving performance benefits.
        
        Args:
            query_hash (str): Unique identifier for the query
            answer (str): Answer to cache for future retrieval
            
        The cache size limit prevents memory issues in long-running systems
        while maintaining performance for frequently asked questions.
        """
        # Store result in cache
        self.query_cache[query_hash] = answer
        
        # Implement simple LRU eviction policy
        if len(self.query_cache) > 1000:  # Cache size limit
            # Remove oldest entries (first 100) to make room
            # This is a simplified LRU implementation
            oldest_keys = list(self.query_cache.keys())[:100]
            for key in oldest_keys:
                del self.query_cache[key]

    def _update_analytics(self, query: str, success: bool) -> None:
        """
        Update search analytics counters for performance monitoring.
        
        This method tracks search patterns and success rates to provide
        insights into system performance and user behavior.
        
        Args:
            query (str): The search query (for future analysis)
            success (bool): Whether the search was successful
            
        Analytics data helps optimize the system and understand usage patterns.
        """
        # Update success/failure counters
        if success:
            self.search_analytics['successful'] += 1
        else:
            self.search_analytics['failed'] += 1
        
        # Update total query counter
        self.search_analytics['total'] += 1

    def _update_usage_stats(self, qa: Dict[str, Any]) -> None:
        """
        Update usage statistics for a specific QA pair.
        
        This method tracks which content is most valuable to users by
        recording access patterns and timestamps.
        
        Args:
            qa (Dict[str, Any]): QA pair dictionary to update
            
        Usage statistics help identify popular content and optimize
        search ranking algorithms.
        """
        # Increment usage counter
        qa['usage_count'] = qa.get('usage_count', 0) + 1
        
        # Record last access time for recency analysis
        qa['last_used'] = datetime.now()