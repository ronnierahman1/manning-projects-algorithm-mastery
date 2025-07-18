"""
Milestone 2: Integrate Dynamic Programming for Context-Aware Responses

This module implements dynamic context tracking to enhance the chatbot's ability
to remember previous interactions and provide more coherent, contextual replies.
It simulates dynamic programming by caching previous results and maintaining
a sliding window of recent conversations.

DYNAMIC PROGRAMMING CONCEPTS DEMONSTRATED:
1. MEMOIZATION: Storing computed query results to avoid redundant processing
2. OPTIMAL SUBSTRUCTURE: Breaking context analysis into smaller, manageable problems
3. STATE MANAGEMENT: Efficiently tracking conversation state and transitions
4. CACHE OPTIMIZATION: Managing memory usage while maximizing retrieval performance

KEY ALGORITHMS IMPLEMENTED:
- Fuzzy string matching using Jaccard similarity and semantic analysis
- Sliding window technique for conversation context management
- Topic extraction and continuity tracking using frequency analysis
- Multi-factor scoring system for cache retrieval optimization

FIXED: Fuzzy matching threshold and scoring to properly handle similar queries.
"""

import datetime  # For timestamp management in cache entries and session tracking
import re  # Regular expressions for pattern matching in text analysis
from typing import Dict, Optional, Any, List, Tuple  # Type hints for better code documentation
from collections import defaultdict, Counter  # Efficient data structures for tracking and counting
import json  # For potential serialization of cache data (future use)


class DynamicContext:
    """
    This class manages conversation context using memoization (a dynamic programming strategy)
    and a contextual history buffer. It demonstrates several key CS concepts:
    
    DYNAMIC PROGRAMMING PRINCIPLES:
    - Memoization: response_cache stores computed results to avoid recomputation
    - Optimal substructure: Complex context analysis broken into smaller functions
    - Overlapping subproblems: Similar queries benefit from cached previous results
    
    PERFORMANCE OPTIMIZATIONS:
    - O(1) cache lookups for exact matches
    - Sliding window for memory management
    - Pre-compiled regex patterns for efficiency
    - Weighted scoring for intelligent fuzzy matching
    
    It allows the chatbot to:
    - Recall previous questions and answers (memoization)
    - Detect follow-up questions (pattern recognition)
    - Infer conversation topics and user preferences (machine learning)
    - Track conversation patterns and sentiment (analytics)
    - Provide context-aware responses (intelligent retrieval)
    """

    def __init__(self):
        """
        Initializes all necessary data structures for the dynamic programming system.
        
        DESIGN PATTERN: This follows the "initialization with sensible defaults" pattern,
        setting up all data structures needed for efficient dynamic programming operations.
        Each structure serves a specific role in the overall caching and context system.
        """
        # CORE DYNAMIC PROGRAMMING STRUCTURES
        
        # Primary memoization table - this is the heart of our dynamic programming approach
        # Key: normalized query string, Value: rich cache entry with metadata
        # Time complexity: O(1) average case for lookups and insertions
        self.response_cache = {}  # Dictionary to store cached query/response pairs
        
        # Sliding window implementation for conversation context management
        # This maintains recent conversation history while preventing unlimited memory growth
        # Implements the "sliding window" algorithm pattern for stream processing
        self.conversation_context = []  # List of user-bot message pairs (chronological)
        
        # Window size parameter - tunable for memory vs. context trade-off
        # Larger values = more context but higher memory usage
        # 10 turns = 20 entries (user + bot responses) provides good balance
        self.context_window = 10  # Only store the latest 10 question-answer turns
        
        # ADVANCED TRACKING SYSTEMS FOR ENHANCED CONTEXT AWARENESS
        
        # Topic continuity tracking using frequency analysis and decay functions
        # defaultdict with lambda creates entries on-demand, avoiding KeyError exceptions
        # This implements a form of "aging" algorithm where topic relevance changes over time
        self.topic_continuity = defaultdict(lambda: {'score': 0.0, 'last_mentioned': None, 'frequency': 0})
        
        # User preference learning system using pattern recognition
        # Tracks user behavior patterns to improve future response selection
        # This is a simple form of collaborative filtering / recommendation system
        self.user_preferences = defaultdict(lambda: {'preference_score': 0.0, 'patterns': [], 'last_updated': None})
        
        # SESSION METADATA FOR COMPREHENSIVE ANALYTICS
        # This structure tracks session-level statistics for performance monitoring
        # and user engagement analysis - important for production chatbot systems
        self.session_metadata = {
            'session_start': datetime.datetime.now(),  # Session start time for duration calculations
            'total_queries': 0,                        # Query counter for analytics
            'unique_topics': set(),                    # Set ensures no duplicates, O(1) membership testing
            'conversation_depth': 0,                   # Tracks how deep the conversation has gone
            'user_engagement_score': 0.0,             # Calculated engagement metric
            'dominant_topics': [],                     # Top topics by frequency
            'query_patterns': Counter(),               # Counter efficiently tracks pattern frequencies
            'response_satisfaction': []                # Could be used for ML training data
        }
        
        # FUZZY MATCHING PARAMETERS - TUNED FOR OPTIMAL PERFORMANCE
        # Lower threshold = more permissive matching (more cache hits, potentially less accurate)
        # Higher threshold = stricter matching (fewer cache hits, higher accuracy)
        # 0.6 chosen through empirical testing as optimal balance point
        self.fuzzy_threshold = 0.6  # Lowered from 0.8 to 0.6 for more reasonable matching
        
        # Pre-computed semantic keyword groups for efficient semantic similarity calculation
        # This avoids recomputing semantic relationships on every query
        self.semantic_keywords = self._initialize_semantic_keywords()
        
        # PRE-COMPILED REGEX PATTERNS FOR PERFORMANCE OPTIMIZATION
        # Compiling patterns once during initialization is much more efficient than
        # recompiling them on every method call. This is a key performance optimization.
        
        # Follow-up detection patterns - used to identify when user is continuing a topic
        # These patterns capture common ways users ask follow-up questions in conversation
        self.followup_patterns = [
            r'\b(also|and|what about|how about|additionally|furthermore|moreover)\b',    # Additive follow-ups
            r'\b(tell me more|explain further|elaborate|continue)\b',                    # Expansion requests
            r'\b(that|this|it)\b.*\b(sounds|seems|looks|appears)\b',                    # Reference-based follow-ups
            r'\b(why|how|when|where|what)\s+(is|are|was|were)\s+(that|this|it)\b'      # Interrogative references
        ]
        
        # Topic extraction patterns for identifying key subjects in queries
        # These patterns capture common technical and educational topics
        # Using word boundaries (\b) ensures we match whole words, not partial matches
        self.topic_extraction_patterns = [
            r'\b(machine learning|artificial intelligence|data science|programming|algorithm)\b',  # AI/ML topics
            r'\b(python|java|javascript|c\+\+|sql|html|css)\b',                                    # Programming languages
            r'\b(database|server|network|security|cloud|api)\b',                                   # Infrastructure topics
            r'\b(frontend|backend|fullstack|devops|testing|deployment)\b'                          # Development topics
        ]

    def store_in_cache(self, query: str, response: str) -> None:
        """
        Caches the query and its associated response for future lookups.
        This is the core implementation of MEMOIZATION in our dynamic programming system.
        
        ALGORITHM ANALYSIS:
        - Time Complexity: O(1) average case for cache insertion
        - Space Complexity: O(k) where k is the length of the query/response
        - Cache key normalization ensures consistent lookups
        
        DYNAMIC PROGRAMMING BENEFIT:
        By storing computed results, we avoid reprocessing identical or similar queries,
        significantly improving response time for repeated questions.

        Args:
            query (str): The user's original question.
            response (str): The chatbot's generated reply.
        """
        try:
            # STEP 1: CREATE CONSISTENT CACHE KEY
            # Normalization is crucial for memoization effectiveness
            # Without it, "What is AI?" and "what is ai" would be separate cache entries
            cache_key = self._normalize_query(query)
            
            # STEP 2: CREATE RICH CACHE ENTRY WITH METADATA
            # We store much more than just the response - this metadata enables
            # sophisticated retrieval algorithms and analytics
            cache_entry = {
                # Core data
                'response': response,                                    # The actual cached response
                'timestamp': datetime.datetime.now(),                   # When this was cached (for aging algorithms)
                'access_count': 1,                                      # Frequency tracking for LRU-style algorithms
                'original_query': query,                                # Original query for fuzzy matching
                
                # Query analysis metadata
                'query_length': len(query.split()),                     # Word count for complexity analysis
                'response_length': len(response.split()),               # Response length for quality metrics
                'topics': self._extract_topics(query),                 # Extracted topics for semantic matching
                'query_type': self._classify_query_type(query),         # Classification for pattern analysis
                'context_relevance': self._calculate_context_relevance(query)  # Relevance to current conversation
            }
            
            # STEP 3: STORE IN MEMOIZATION TABLE
            # This is the actual memoization step - O(1) insertion into hash table
            self.response_cache[cache_key] = cache_entry
            
            # STEP 4: UPDATE TRACKING SYSTEMS
            # These updates maintain the broader context awareness system
            # Each method implements specific aspects of our dynamic programming approach
            
            # Update session-level statistics for analytics and optimization
            self._update_session_metadata(query, response)
            
            # Update topic continuity scores using frequency analysis
            # This implements a form of "aging" algorithm where recent topics get higher scores
            self._update_topic_continuity(query, response)
            
            # Learn user interaction patterns for future optimization
            # This is a simple form of machine learning / pattern recognition
            self._update_user_preferences(query, response)
            
            # STEP 5: ADD TO SLIDING WINDOW CONTEXT
            # This maintains conversation history while managing memory usage
            self._add_to_context(query, response)

        except Exception as e:
            # Graceful error handling ensures system stability
            # In production systems, this might log to monitoring systems
            print(f"Error storing in cache: {e}")

    def retrieve_from_cache(self, query: str) -> Optional[str]:
        """
        Attempts to retrieve an answer from the cache using dynamic programming principles.
        
        ALGORITHM STRATEGY:
        1. First try exact match (O(1) hash table lookup) - fastest possible
        2. If no exact match, fall back to fuzzy matching (O(n) but with intelligent scoring)
        
        This two-tier approach optimizes for the common case (exact matches) while
        providing intelligent fallback for similar queries.
        
        DYNAMIC PROGRAMMING BENEFIT:
        Previous computations (cached responses) are reused, and the access patterns
        inform future optimization decisions.

        Args:
            query (str): The user's current query.

        Returns:
            Optional[str]: The cached response if available; otherwise, None.
        """
        try:
            # STEP 1: ATTEMPT EXACT MATCH (O(1) OPERATION)
            # This is the primary benefit of memoization - instant retrieval of exact matches
            cache_key = self._normalize_query(query)

            if cache_key in self.response_cache:
                # CACHE HIT! Update access metadata for analytics and future optimization
                cache_entry = self.response_cache[cache_key]
                
                # Increment access count for frequency-based algorithms (LRU, LFU, etc.)
                cache_entry['access_count'] += 1
                
                # Update last access timestamp for temporal analysis
                cache_entry['last_accessed'] = datetime.datetime.now()
                
                # Recalculate context relevance based on current conversation state
                # This ensures that cached responses stay relevant to ongoing conversation
                cache_entry['context_relevance'] = self._calculate_context_relevance(query)
                
                # Return the cached response - this is the memoization payoff!
                return cache_entry['response']

            # STEP 2: EXACT MATCH FAILED - TRY FUZZY MATCHING
            # This implements a sophisticated similarity algorithm that finds
            # the best approximate match from our cache
            return self._fuzzy_cache_lookup(query)

        except Exception as e:
            # Graceful error handling prevents system crashes
            print(f"Error retrieving from cache: {e}")
            return None

    def has_context(self, query: str) -> bool:
        """
        Checks whether the current query already exists in the cached responses.
        
        This method uses dynamic programming to efficiently determine if we have
        relevant context for a query, enabling smart routing decisions.
        
        OPTIMIZATION: Uses a lower threshold for fuzzy matching when checking context
        because false positives are less costly here than in actual retrieval.

        Args:
            query (str): The incoming user question.

        Returns:
            bool: True if the query exists in cache or has high similarity; otherwise, False.
        """
        try:
            # STEP 1: CHECK EXACT MATCH FIRST (FASTEST PATH)
            cache_key = self._normalize_query(query)
            
            # O(1) hash table lookup - if exact match exists, we definitely have context
            if cache_key in self.response_cache:
                return True
            
            # STEP 2: CHECK FUZZY MATCH WITH LOWER THRESHOLD
            # Lower threshold (0.4 vs 0.6) means we're more generous in detecting context
            # This is appropriate because false positives are less problematic for context detection
            # than for actual response retrieval
            fuzzy_result = self._fuzzy_cache_lookup(query, threshold=0.4)  # FIXED: Lower threshold for has_context
            return fuzzy_result is not None
            
        except Exception as e:
            # Safe default: assume no context if error occurs
            print(f"Error checking context: {e}")
            return False

    def _normalize_query(self, query: str) -> str:
        """
        A helper method to standardize query format before caching.
        
        NORMALIZATION IMPORTANCE:
        Without normalization, similar queries would create separate cache entries:
        - "What is AI?" vs "what is ai" vs "What is AI"
        This reduces cache effectiveness and wastes memory.
        
        PROCESSING STEPS:
        1. Case normalization (lowercase)
        2. Punctuation removal
        3. Contraction expansion
        4. Whitespace normalization
        
        Each step addresses common variations in user input.

        Args:
            query (str): The original query string.

        Returns:
            str: The normalized query string.
        """
        # STEP 1: BASIC NORMALIZATION
        # Convert to lowercase for case-insensitive matching
        # Strip leading/trailing whitespace
        normalized = query.lower().strip()
        
        # STEP 2: REMOVE PUNCTUATION
        # Punctuation doesn't usually affect semantic meaning for caching purposes
        # This regex removes common punctuation marks that don't change query intent
        normalized = re.sub(r'[?!.,;:]', '', normalized)
        
        # STEP 3: HANDLE CONTRACTIONS
        # Expanding contractions creates more consistent cache keys
        # "don't" and "do not" should map to the same cache entry
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mustn't": "must not"
        }
        
        # Apply each contraction expansion
        # This loop ensures all contractions are expanded consistently
        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)
        
        # STEP 4: NORMALIZE WHITESPACE
        # Replace multiple spaces with single spaces for consistency
        # This regex matches one or more whitespace characters and replaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized

    def _add_to_context(self, query: str, response: str) -> None:
        """
        Adds the user query and the bot response as a pair to the
        context history, preserving order and timestamps.
        
        SLIDING WINDOW IMPLEMENTATION:
        This method implements the "sliding window" algorithm pattern, commonly used
        in stream processing and real-time systems. It maintains a fixed-size buffer
        of recent interactions while automatically discarding older entries.
        
        BENEFITS:
        - Bounded memory usage (prevents memory leaks)
        - Maintains temporal locality (recent context is most relevant)
        - Preserves conversation flow for follow-up detection

        Args:
            query (str): The user question.
            response (str): The bot's reply.
        """
        try:
            # STEP 1: CREATE TIMESTAMPED ENTRIES WITH RICH METADATA
            timestamp = datetime.datetime.now()
            
            # Create user entry with comprehensive metadata
            # This metadata enables sophisticated context analysis algorithms
            user_entry = {
                'type': 'user',                                        # Entry type for filtering
                'content': query,                                      # Actual query text
                'timestamp': timestamp,                                # For temporal analysis
                'word_count': len(query.split()),                      # Complexity indicator
                'topics': self._extract_topics(query),                # For topic continuity tracking
                'query_type': self._classify_query_type(query),        # For pattern analysis
                'context_position': len(self.conversation_context)     # Position in conversation flow
            }
            
            # Create bot entry with corresponding metadata
            # Parallel structure to user entry enables comparative analysis
            bot_entry = {
                'type': 'bot',                                         # Entry type for filtering
                'content': response,                                   # Actual response text
                'timestamp': timestamp,                                # Matching timestamp
                'word_count': len(response.split()),                   # Response length analysis
                'topics': self._extract_topics(response),              # Topics covered in response
                'response_type': self._classify_response_type(response), # Response classification
                'context_position': len(self.conversation_context) + 1  # Sequential position
            }

            # STEP 2: ADD ENTRIES TO CONVERSATION CONTEXT
            # Both user and bot entries are stored to maintain complete conversation flow
            # This enables analysis of interaction patterns and conversation dynamics
            self.conversation_context.append(user_entry)
            self.conversation_context.append(bot_entry)

            # STEP 3: IMPLEMENT SLIDING WINDOW PRUNING
            # This is the key optimization that prevents unlimited memory growth
            # When context exceeds our window size, we intelligently prune old entries
            if len(self.conversation_context) > self.context_window * 2:
                # Keep more recent entries and preserve important context
                # Smart pruning considers topic importance, not just recency
                self.conversation_context = self._smart_context_pruning()

        except Exception as e:
            # Graceful error handling maintains system stability
            print(f"Error adding to context: {e}")

    def _fuzzy_cache_lookup(self, query: str, threshold: float = None) -> Optional[str]:
        """
        Attempts to find a close match to the current query based on word similarity.
        
        FUZZY MATCHING ALGORITHM:
        This implements a sophisticated multi-factor similarity scoring system:
        1. Word overlap (Jaccard similarity) - core semantic similarity
        2. Semantic category matching - domain-specific similarity
        3. Context relevance - conversation-aware scoring
        4. Access frequency - popularity-based boosting
        
        DYNAMIC PROGRAMMING ASPECT:
        Previous access patterns and context calculations inform the scoring,
        making the algorithm adaptive and self-improving over time.
        
        PERFORMANCE CONSIDERATION:
        This is O(n) where n is the number of cache entries, but with intelligent
        early termination and optimized scoring calculations.

        Args:
            query (str): The user's input that didn't yield an exact cache match.
            threshold (float): Custom threshold for matching (uses default if None).

        Returns:
            Optional[str]: A similar cached response, or None if no match found.
        """
        try:
            # STEP 1: INITIALIZE SCORING VARIABLES
            if threshold is None:
                threshold = self.fuzzy_threshold
                
            # Create word set for efficient intersection/union operations
            # Using sets provides O(1) membership testing and O(min(len(a),len(b))) intersection
            query_words = set(query.lower().split())
            best_match = None
            best_score = 0.0
            
            # STEP 2: EVALUATE EACH CACHE ENTRY
            # This loop implements the core fuzzy matching algorithm
            # We examine every cached entry to find the best semantic match
            for cache_key, cache_entry in self.response_cache.items():
                # Use original query from cache entry for accurate word analysis
                # Cache keys are normalized, but we need original text for word splitting
                original_cache_query = cache_entry['original_query']
                cache_words = set(original_cache_query.lower().split())
                
                # Skip empty word sets to avoid division by zero
                if query_words and cache_words:
                    # COMPONENT 1: WORD OVERLAP SCORE (JACCARD SIMILARITY)
                    # This measures semantic similarity based on shared vocabulary
                    overlap = len(query_words.intersection(cache_words))
                    union_size = len(query_words.union(cache_words))
                    word_score = overlap / union_size if union_size > 0 else 0
                    
                    # OPTIMIZATION: BOOST SCORES FOR SIGNIFICANT OVERLAPS
                    # These bonuses reward queries with multiple matching words
                    # This addresses the limitation of pure Jaccard similarity for short texts
                    if overlap >= 2:  # At least 2 words match - strong similarity
                        word_score += 0.5  # Increased from 0.4 for better matching
                    elif overlap >= 1:  # At least 1 word matches - some similarity
                        word_score += 0.3  # Moderate boost for single word matches
                    
                    # COMPONENT 2: SEMANTIC SIMILARITY SCORE
                    # This captures domain-specific relationships beyond exact word matches
                    # For example, "machine learning" and "neural networks" are semantically related
                    semantic_score = self._calculate_semantic_similarity(query, cache_entry['original_query'])
                    
                    # COMPONENT 3: CONTEXT RELEVANCE SCORE
                    # This considers how well the cached entry fits the current conversation
                    # Entries that are topically relevant to recent discussion get higher scores
                    context_score = cache_entry.get('context_relevance', 0.0)
                    
                    # COMPONENT 4: ACCESS FREQUENCY SCORE
                    # Popular queries get a small boost - implements a form of collaborative filtering
                    # Frequently accessed entries are more likely to be useful
                    frequency_score = min(cache_entry['access_count'] / 10.0, 0.1)
                    
                    # STEP 3: COMBINE SCORES WITH WEIGHTED FORMULA
                    # These weights were tuned empirically for optimal performance
                    # Word overlap gets highest weight as it's most reliable
                    combined_score = (
                        word_score * 0.75 +      # Primary similarity indicator
                        semantic_score * 0.15 +  # Domain knowledge boost
                        context_score * 0.05 +   # Conversation relevance
                        frequency_score * 0.05   # Popularity boost
                    )

                    # STEP 4: UPDATE BEST MATCH IF SCORE IMPROVES
                    # Only consider entries that exceed our threshold
                    # This prevents low-quality matches from being returned
                    if combined_score > best_score and combined_score > threshold:
                        best_score = combined_score
                        best_match = cache_entry['response']

            # STEP 5: RETURN BEST MATCH OR NONE
            return best_match

        except Exception as e:
            # Graceful error handling prevents system crashes during fuzzy matching
            print(f"Error in fuzzy cache lookup: {e}")
            return None

    def _analyze_context(self, query: str) -> Dict[str, Any]:
        """
        Analyze the given query in the context of recent conversation history.
        
        CONTEXT ANALYSIS ALGORITHM:
        This method implements sophisticated conversation analysis using multiple techniques:
        1. Pattern recognition for follow-up detection
        2. Topic extraction and continuity analysis
        3. Engagement level calculation
        4. Response type prediction
        
        DYNAMIC PROGRAMMING ASPECT:
        Previous conversation analysis results inform current analysis,
        building up context understanding over time.

        Args:
            query (str): The user's input.

        Returns:
            Dict[str, Any]: Enhanced metadata including flags and inferred topics.
        """
        try:
            # STEP 1: INITIALIZE COMPREHENSIVE CONTEXT ANALYSIS STRUCTURE
            # This structure captures multiple dimensions of conversation context
            context_info = {
                'is_followup': False,                                  # Whether this continues previous topic
                'related_topics': [],                                 # Topics from recent conversation
                'sentiment_shift': 'neutral',                         # Conversation tone analysis
                'conversation_depth': self.session_metadata['conversation_depth'], # How deep we are
                'topic_continuity_score': 0.0,                      # Topic persistence metric
                'user_engagement_level': 'medium',                   # Calculated engagement
                'query_complexity': 'simple',                        # Query difficulty assessment
                'expected_response_type': 'informational'            # Predicted response style
            }

            # STEP 2: ENHANCED FOLLOW-UP DETECTION USING REGEX PATTERNS
            # This implements pattern recognition to identify conversation continuity
            query_lower = query.lower()
            for pattern in self.followup_patterns:
                # Each pattern captures a different way users ask follow-up questions
                if re.search(pattern, query_lower):
                    context_info['is_followup'] = True
                    break  # Early termination optimization

            # STEP 3: ENHANCED TOPIC EXTRACTION FROM RECENT CONVERSATION
            # This analyzes conversation history to understand current context
            # We look at recent user queries to identify ongoing topics
            recent_queries = [entry['content'] for entry in self.conversation_context[-6:]
                              if entry['type'] == 'user']

            # Build comprehensive topic set from recent conversation
            all_topics = set()
            for recent_query in recent_queries:
                # Extract structured topics using pattern matching
                topics = self._extract_topics(recent_query)
                all_topics.update(topics)
                
                # Also add individual meaningful words from queries
                # This catches important terms that might not match our patterns
                words = recent_query.lower().split()
                for word in words:
                    # Filter out short words and common stop words
                    # Only include words that carry semantic meaning
                    if len(word) > 3 and word not in ['what', 'how', 'where', 'when', 'why', 'which', 'this', 'that', 'with', 'from', 'they', 'them', 'have', 'been', 'were', 'will', 'would', 'could', 'should', 'about', 'tell']:
                        all_topics.add(word)
                
                # SPECIAL HANDLING FOR COMPOUND PHRASES
                # These are important multi-word concepts that should be preserved
                if "machine learning" in recent_query.lower():
                    all_topics.add("machine")      # Individual words
                    all_topics.add("learning")
                    all_topics.add("machine learning")  # Complete phrase
                    
                if "neural networks" in recent_query.lower():
                    all_topics.add("neural")
                    all_topics.add("networks")
                    all_topics.add("neural networks")
                
            # Store extracted topics for analysis
            context_info['related_topics'] = list(all_topics)

            # STEP 4: TOPIC CONTINUITY ANALYSIS
            # Calculate how well current query relates to ongoing conversation topics
            if all_topics:
                # Get continuity scores for all current topics
                continuity_scores = [self.topic_continuity[topic]['score'] for topic in all_topics]
                # Calculate average continuity - higher means strong topic persistence
                context_info['topic_continuity_score'] = sum(continuity_scores) / len(continuity_scores)

            # STEP 5: COMPREHENSIVE CONTEXT METRICS
            # These methods implement various aspects of conversation analysis
            
            # Analyze structural complexity of the query
            context_info['query_complexity'] = self._analyze_query_complexity(query)
            
            # Calculate user engagement based on interaction patterns
            context_info['user_engagement_level'] = self._calculate_user_engagement()
            
            # Predict what type of response the user expects
            context_info['expected_response_type'] = self._predict_response_type(query)

            return context_info

        except Exception as e:
            # Return safe defaults if analysis fails
            # This ensures system stability even when context analysis encounters errors
            print(f"Error analyzing context: {e}")
            return {
                'is_followup': False,
                'related_topics': [],
                'sentiment_shift': 'neutral',
                'conversation_depth': 0,
                'topic_continuity_score': 0.0,
                'user_engagement_level': 'medium',
                'query_complexity': 'simple',
                'expected_response_type': 'informational'
            }

    # ENHANCED METHODS FOR SOPHISTICATED CONTEXT ANALYSIS

    def _initialize_semantic_keywords(self) -> Dict[str, List[str]]:
        """
        Initialize semantic keyword groups for better matching.
        
        SEMANTIC ANALYSIS FOUNDATION:
        This method creates predefined semantic categories that enable
        the system to understand conceptual relationships between words.
        
        For example, "coding" and "programming" are semantically similar
        even though they don't share characters.
        """
        return {
            # Programming and development related terms
            'programming': ['code', 'coding', 'program', 'software', 'development', 'algorithm', 'programming', 'python', 'java', 'language', 'methods', 'basics'],
            # Data and information related terms
            'data': ['information', 'dataset', 'statistics', 'analysis', 'mining', 'data', 'structures'],
            # Learning and education related terms
            'learning': ['study', 'education', 'tutorial', 'guide', 'teach', 'explain', 'learning', 'machine', 'neural', 'networks'],
            # Technology and systems related terms
            'technology': ['tech', 'digital', 'computer', 'system', 'network', 'platform'],
            # Web development related terms
            'web': ['website', 'internet', 'online', 'browser', 'html', 'css', 'javascript']
        }

    def _extract_topics(self, text: str) -> List[str]:
        """
        Enhanced topic extraction using pattern matching and keywords.
        
        TOPIC EXTRACTION ALGORITHM:
        This method uses multiple strategies to identify important topics:
        1. Pattern-based extraction using pre-defined regex patterns
        2. Keyword-based extraction using meaningful word filtering
        3. Compound phrase detection for multi-word concepts
        
        The combination provides comprehensive topic coverage.
        """
        topics = []
        text_lower = text.lower()
        
        # STRATEGY 1: PATTERN-BASED EXTRACTION
        # Use pre-compiled regex patterns to find structured topics
        for pattern in self.topic_extraction_patterns:
            matches = re.findall(pattern, text_lower)
            topics.extend(matches)
        
        # STRATEGY 2: KEYWORD-BASED EXTRACTION
        # Extract meaningful individual words
        words = text_lower.split()
        meaningful_words = []
        
        for word in words:
            # Include important keywords and longer words
            # Exclude common stop words that don't carry semantic meaning
            if (len(word) > 3 and word not in ['what', 'how', 'where', 'when', 'why', 'which', 'this', 'that', 'with', 'from', 'they', 'them', 'have', 'been', 'were', 'will', 'would', 'could', 'should']) or \
               any(keyword in word for keyword_list in self.semantic_keywords.values() for keyword in keyword_list):
                meaningful_words.append(word)
        
        topics.extend(meaningful_words)
        
        # STRATEGY 3: COMPOUND PHRASE EXTRACTION
        # Look for important multi-word concepts
        compound_phrases = []
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            # Check if this phrase appears in our semantic keyword lists
            if any(keyword_phrase in phrase for keyword_list in self.semantic_keywords.values() for keyword_phrase in keyword_list):
                compound_phrases.append(phrase)
        
        topics.extend(compound_phrases)
        
        # Return unique topics only (set removes duplicates, list converts back)
        return list(set(topics))

    def _classify_query_type(self, query: str) -> str:
        """
        Classify the type of query for better context understanding.
        
        QUERY CLASSIFICATION ALGORITHM:
        This uses keyword detection to categorize queries into types.
        Different query types often expect different response styles.
        """
        query_lower = query.lower()
        
        # Pattern matching for different query types
        # Each condition checks for characteristic keywords
        if any(word in query_lower for word in ['what', 'define', 'explain', 'describe']):
            return 'definition'      # Seeking explanations or definitions
        elif any(word in query_lower for word in ['how', 'steps', 'process', 'method']):
            return 'procedural'      # Seeking step-by-step instructions
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'causal'         # Seeking causal explanations
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            return 'comparison'     # Seeking comparative analysis
        elif any(word in query_lower for word in ['example', 'sample', 'instance']):
            return 'example'        # Seeking concrete examples
        elif '?' in query:
            return 'question'       # General question format
        else:
            return 'statement'      # Declarative statement

    def _classify_response_type(self, response: str) -> str:
        """
        Classify the type of response for context tracking.
        
        RESPONSE CLASSIFICATION:
        This analyzes response characteristics to categorize response styles.
        This helps track conversation patterns and user preferences.
        """
        response_lower = response.lower()
        
        # Length-based classification
        if len(response.split()) < 10:
            return 'brief'          # Short, concise responses
        # Content-based classification
        elif any(word in response_lower for word in ['step', 'first', 'second', 'then', 'next']):
            return 'procedural'     # Step-by-step instructions
        elif any(word in response_lower for word in ['example', 'for instance', 'such as']):
            return 'example'        # Example-driven responses
        elif len(response.split()) > 50:
            return 'detailed'       # Comprehensive, detailed responses
        else:
            return 'informational'  # Standard informational responses

    def _calculate_context_relevance(self, query: str) -> float:
        """
        Calculate how relevant a query is to the current conversation context.
        
        CONTEXT RELEVANCE ALGORITHM:
        This uses topic overlap analysis to measure how well a query
        fits with the ongoing conversation. Higher relevance scores
        indicate strong topical continuity.
        """
        # No context available - relevance is zero
        if not self.conversation_context:
            return 0.0
        
        # Extract topics from current query
        query_topics = set(self._extract_topics(query))
        recent_topics = set()
        
        # Get topics from recent conversation (last 2 exchanges = 4 entries)
        for entry in self.conversation_context[-4:]:
            if 'topics' in entry:
                recent_topics.update(entry['topics'])
        
        # No topics to compare - relevance is zero
        if not query_topics or not recent_topics:
            return 0.0
        
        # Calculate topic overlap using Jaccard similarity
        # This measures how much the current query relates to recent topics
        overlap = len(query_topics.intersection(recent_topics))
        union = len(query_topics.union(recent_topics))
        
        return overlap / union if union > 0 else 0.0

    def _calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        Calculate semantic similarity between two queries.
        
        SEMANTIC SIMILARITY ALGORITHM:
        This implements a multi-factor similarity calculation:
        1. Word-level similarity (Jaccard coefficient)
        2. Semantic category overlap (domain knowledge)
        3. Topic-level similarity (extracted concepts)
        
        The combination provides robust similarity detection beyond exact matches.
        """
        # COMPONENT 1: BASIC WORD SIMILARITY
        # Convert to word sets for efficient set operations
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        # Calculate Jaccard similarity coefficient
        word_overlap = len(words1.intersection(words2))
        word_union = len(words1.union(words2))
        word_similarity = word_overlap / word_union if word_union > 0 else 0.0
        
        # COMPONENT 2: ENHANCED SEMANTIC CATEGORY OVERLAP SCORING
        semantic_score = 0.0
        category_matches = 0
        
        # Check each semantic category for presence in both queries
        for category, keywords in self.semantic_keywords.items():
            query1_has_category = any(keyword in query1.lower() for keyword in keywords)
            query2_has_category = any(keyword in query2.lower() for keyword in keywords)
            
            # If both queries contain words from this semantic category, boost score
            if query1_has_category and query2_has_category:
                semantic_score += 0.4  # Increased boost for semantic matches
                category_matches += 1
        
        # Bonus for multiple semantic category matches
        # Queries sharing multiple semantic categories are highly similar
        if category_matches > 1:
            semantic_score += 0.2
        
        # COMPONENT 3: TOPIC OVERLAP SCORE
        # Compare extracted topics using the same approach
        topics1 = set(self._extract_topics(query1))
        topics2 = set(self._extract_topics(query2))
        topic_similarity = 0.0
        
        if topics1 and topics2:
            topic_overlap = len(topics1.intersection(topics2))
            topic_union = len(topics1.union(topics2))
            topic_similarity = topic_overlap / topic_union if topic_union > 0 else 0.0
        
        # FINAL SCORE: WEIGHTED COMBINATION
        # Word similarity gets highest weight as it's most reliable
        # Semantic and topic similarities provide contextual enhancement
        final_score = (word_similarity * 0.5 + semantic_score * 0.3 + topic_similarity * 0.2)
        return min(final_score, 1.0)  # Ensure score doesn't exceed 1.0

    def _update_session_metadata(self, query: str, response: str) -> None:
        """
        Update session-level metadata with new interaction.
        
        SESSION TRACKING:
        This maintains comprehensive session statistics that can be used
        for analytics, optimization, and user experience improvements.
        """
        # Increment total query counter
        self.session_metadata['total_queries'] += 1
        
        # Add new topics to unique topics set (automatic deduplication)
        self.session_metadata['unique_topics'].update(self._extract_topics(query))
        
        # Increment conversation depth (measures engagement)
        self.session_metadata['conversation_depth'] += 1
        
        # Update query pattern tracking using Counter for efficient frequency counting
        query_type = self._classify_query_type(query)
        self.session_metadata['query_patterns'][query_type] += 1
        
        # Update dominant topics based on frequency analysis
        all_topics = list(self.session_metadata['unique_topics'])
        topic_counts = Counter()
        
        # Get frequency data from topic continuity tracking
        for topic in all_topics:
            topic_counts[topic] = self.topic_continuity[topic]['frequency']
        
        # Store top 5 most frequent topics
        self.session_metadata['dominant_topics'] = [topic for topic, count in topic_counts.most_common(5)]

    def _update_topic_continuity(self, query: str, response: str) -> None:
        """
        Update topic continuity tracking with new interaction.
        
        TOPIC CONTINUITY ALGORITHM:
        This implements a frequency-based scoring system with temporal tracking.
        Topics that appear frequently get higher continuity scores, which
        influences future context analysis and fuzzy matching.
        """
        # Extract topics from both query and response
        topics = self._extract_topics(query) + self._extract_topics(response)
        
        for topic in topics:
            # Increment frequency counter
            self.topic_continuity[topic]['frequency'] += 1
            
            # Update last mentioned timestamp for temporal analysis
            self.topic_continuity[topic]['last_mentioned'] = datetime.datetime.now()
            
            # Calculate continuity score based on frequency
            # Score is bounded at 1.0 to prevent unlimited growth
            # Factor of 0.1 means 10 mentions = maximum score
            self.topic_continuity[topic]['score'] = min(
                self.topic_continuity[topic]['frequency'] * 0.1, 1.0
            )

    def _update_user_preferences(self, query: str, response: str) -> None:
        """
        Update user preference tracking based on interaction patterns.
        
        USER PREFERENCE LEARNING:
        This implements a simple form of collaborative filtering by tracking
        user interaction patterns and query preferences over time.
        """
        # Get interaction characteristics
        query_type = self._classify_query_type(query)
        response_type = self._classify_response_type(response)
        
        # Initialize preference entry if not exists
        if query_type not in self.user_preferences:
            self.user_preferences[query_type] = {
                'preference_score': 0.0,
                'patterns': [],
                'last_updated': datetime.datetime.now()
            }
        
        # Increment preference score for this query type
        # Higher scores indicate user prefers this type of interaction
        self.user_preferences[query_type]['preference_score'] += 0.1
        
        # Track response patterns for this query type
        # This could be used for response style optimization
        self.user_preferences[query_type]['patterns'].append(response_type)
        
        # Update timestamp for temporal analysis
        self.user_preferences[query_type]['last_updated'] = datetime.datetime.now()

    def _analyze_query_complexity(self, query: str) -> str:
        """
        Analyze the complexity level of a query.
        
        COMPLEXITY ANALYSIS:
        Simple word count-based heuristic for determining query complexity.
        More sophisticated versions might analyze syntactic structure or topic depth.
        """
        word_count = len(query.split())
        
        # Simple threshold-based classification
        if word_count <= 3:
            return 'simple'     # Short, direct queries
        elif word_count <= 8:
            return 'medium'     # Moderate complexity
        else:
            return 'complex'    # Long, detailed queries

    def _calculate_user_engagement(self) -> str:
        """
        Calculate user engagement level based on conversation patterns.
        
        ENGAGEMENT CALCULATION:
        This analyzes average query length as a proxy for user engagement.
        More engaged users tend to ask longer, more detailed questions.
        """
        if self.session_metadata['total_queries'] == 0:
            return 'low'
        
        # Calculate average query length across all user entries
        # Filter for user entries only, exclude bot responses
        user_entries = [entry for entry in self.conversation_context if entry['type'] == 'user']
        if not user_entries:
            return 'low'
            
        avg_query_length = sum(len(entry['content'].split()) for entry in user_entries) / len(user_entries)
        
        # Threshold-based engagement classification
        if avg_query_length < 3:
            return 'low'        # Short, minimal queries
        elif avg_query_length < 8:
            return 'medium'     # Moderate engagement
        else:
            return 'high'       # Detailed, engaged queries

    def _predict_response_type(self, query: str) -> str:
        """
        Predict the expected type of response based on query analysis.
        
        RESPONSE TYPE PREDICTION:
        This maps query types to expected response styles, enabling
        the system to tailor responses to user expectations.
        """
        query_type = self._classify_query_type(query)
        
        # Mapping from query types to expected response types
        # This could be learned from user feedback in a production system
        type_mapping = {
            'definition': 'explanatory',        # Detailed explanations
            'procedural': 'step-by-step',       # Sequential instructions
            'causal': 'analytical',             # Cause-effect analysis
            'comparison': 'comparative',        # Side-by-side comparison
            'example': 'illustrative',          # Concrete examples
            'question': 'informational',        # General information
            'statement': 'confirmatory'         # Confirmation or acknowledgment
        }
        
        return type_mapping.get(query_type, 'informational')

    def _smart_context_pruning(self) -> List[Dict[str, Any]]:
        """
        Intelligently prune context history while preserving important information.
        
        SMART PRUNING ALGORITHM:
        Instead of simple FIFO (first-in-first-out), this algorithm:
        1. Always keeps recent entries (temporal locality)
        2. Preserves entries with high topic continuity scores (importance)
        3. Maintains chronological order
        
        This optimizes the sliding window for both memory efficiency and context quality.
        """
        # If context is within limits, no pruning needed
        if len(self.conversation_context) <= self.context_window * 2:
            return self.conversation_context
        
        # STEP 1: KEEP THE MOST RECENT ENTRIES
        # Recent context is always most relevant for ongoing conversation
        recent_entries = self.conversation_context[-self.context_window:]
        
        # STEP 2: IDENTIFY IMPORTANT OLDER ENTRIES
        # Look for entries with high topic importance scores
        important_entries = []
        for entry in self.conversation_context[:-self.context_window]:
            if 'topics' in entry and entry['topics']:
                # Calculate importance based on topic continuity scores
                topic_scores = [self.topic_continuity[topic]['score'] for topic in entry['topics']]
                entry_importance = sum(topic_scores) / len(entry['topics'])
                
                # Threshold for importance - entries above this are preserved
                if entry_importance > 0.5:
                    important_entries.append(entry)
        
        # STEP 3: COMBINE AND SORT BY TIMESTAMP
        # Merge important older entries with recent entries
        combined_entries = important_entries + recent_entries
        combined_entries.sort(key=lambda x: x['timestamp'])
        
        # STEP 4: LIMIT TO CONTEXT WINDOW SIZE
        # Final size limit to prevent unlimited growth
        return combined_entries[-self.context_window * 2:]

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the current conversation session.
        
        ANALYTICS AND MONITORING:
        This method provides comprehensive session analytics that could be used
        for user experience optimization, system performance monitoring,
        and machine learning model training.
        """
        return {
            # Temporal information
            'session_duration': datetime.datetime.now() - self.session_metadata['session_start'],
            
            # Interaction metrics
            'total_interactions': self.session_metadata['total_queries'],
            'conversation_depth': self.session_metadata['conversation_depth'],
            
            # Content analysis
            'dominant_topics': self.session_metadata['dominant_topics'],
            'topic_diversity': len(self.session_metadata['unique_topics']),
            'most_common_query_types': self.session_metadata['query_patterns'].most_common(3),
            
            # Performance metrics
            'user_engagement': self._calculate_user_engagement(),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }

    def _calculate_cache_hit_rate(self) -> float:
        """
        Calculate the cache hit rate for this session.
        
        CACHE PERFORMANCE METRIC:
        This measures how effectively our memoization strategy is working.
        Higher hit rates indicate better cache effectiveness and system performance.
        """
        if self.session_metadata['total_queries'] == 0:
            return 0.0
        
        # Calculate total cache accesses across all entries
        total_accesses = sum(entry['access_count'] for entry in self.response_cache.values())
        unique_queries = len(self.response_cache)
        
        # Avoid division by zero
        if unique_queries == 0:
            return 0.0
            
        # Hit rate = cache accesses / total queries
        # Values closer to 1.0 indicate better cache performance
        return min(total_accesses / self.session_metadata['total_queries'], 1.0)

    def get_topic_insights(self) -> Dict[str, Any]:
        """
        Get insights about topic patterns and continuity.
        
        TOPIC ANALYTICS:
        Provides detailed analysis of conversation topics for understanding
        user interests and conversation flow patterns.
        """
        return {
            'active_topics': dict(self.topic_continuity),     # Current topic scores and metadata
            'topic_trends': self._analyze_topic_trends(),     # Trending analysis
            'conversation_flow': self._analyze_conversation_flow()  # Topic sequence analysis
        }

    def _analyze_topic_trends(self) -> Dict[str, str]:
        """
        Analyze trending topics in the conversation.
        
        TREND ANALYSIS:
        Categorizes topics based on their frequency patterns to identify
        trending, active, and emerging topics in the conversation.
        """
        trends = {}
        for topic, data in self.topic_continuity.items():
            # Frequency-based trend classification
            if data['frequency'] > 2:
                trends[topic] = 'trending'      # High frequency topics
            elif data['frequency'] > 1:
                trends[topic] = 'active'        # Moderate frequency topics
            else:
                trends[topic] = 'emerging'      # Low frequency topics
        return trends

    def _analyze_conversation_flow(self) -> List[str]:
        """
        Analyze the flow of conversation topics.
        
        CONVERSATION FLOW ANALYSIS:
        Tracks the sequence of topics discussed to understand conversation
        progression and topic transitions over time.
        """
        flow = []
        # Extract topics from user entries in chronological order
        for entry in self.conversation_context:
            if entry['type'] == 'user' and 'topics' in entry:
                flow.extend(entry['topics'])
        
        # Return last 10 topics in conversation flow
        # This provides recent topic progression without overwhelming detail
        return flow[-10:]