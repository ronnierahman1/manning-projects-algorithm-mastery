"""
Milestone 2: Integrate Dynamic Programming for Context-Aware Responses - STARTER CODE

This module implements dynamic context tracking to enhance the chatbot's ability
to remember previous interactions and provide more coherent, contextual replies.
It simulates dynamic programming by caching previous results and maintaining
a sliding window of recent conversations.

LEARNING OBJECTIVES:
- Implement dynamic programming concepts through memoization and caching
- Build context-aware response systems that remember conversation history
- Create fuzzy matching algorithms for similar query detection
- Develop sliding window techniques for managing conversation context
- Practice cache optimization and retrieval strategies

KEY CONCEPTS TO IMPLEMENT:
1. Memoization - storing computed results to avoid redundant calculations
2. Cache management - efficient storage and retrieval of query-response pairs
3. Fuzzy matching - finding similar queries using word overlap and semantic similarity
4. Context sliding window - maintaining recent conversation history within limits
5. Topic tracking - monitoring conversation topics and continuity over time
"""

import datetime
import re
from typing import Dict, Optional, Any, List, Tuple
from collections import defaultdict, Counter
import json


class DynamicContext:
    """
    This class manages conversation context using memoization (a dynamic programming strategy)
    and a contextual history buffer. It allows the chatbot to:
    - Recall previous questions and answers
    - Detect follow-up questions
    - Infer conversation topics and user preferences
    - Track conversation patterns and sentiment
    - Provide context-aware responses
    
    IMPLEMENTATION GUIDE:
    This class demonstrates dynamic programming through:
    1. Memoization - caching query results to avoid recomputation
    2. Optimal substructure - breaking context analysis into smaller problems
    3. State management - tracking conversation state efficiently
    4. Cache optimization - managing memory and retrieval performance
    """

    def __init__(self):
        """
        Initializes all necessary data structures for dynamic context management.
        
        TODO: UNDERSTAND THE DATA STRUCTURES
        Each data structure serves a specific purpose in the dynamic programming approach:
        - response_cache: The main memoization table for query-response pairs
        - conversation_context: Sliding window of recent interactions
        - topic_continuity: Dynamic tracking of topic frequency and relevance
        - user_preferences: Adaptive learning of user interaction patterns
        """
        # Core dynamic programming structures
        self.response_cache = {}  # Dictionary to store cached query/response pairs (memoization table)
        self.conversation_context = []  # List of user-bot message pairs (sliding window)
        self.context_window = 10  # Only store the latest 10 question-answer turns
        
        # Enhanced tracking structures
        self.topic_continuity = defaultdict(lambda: {'score': 0.0, 'last_mentioned': None, 'frequency': 0})
        self.user_preferences = defaultdict(lambda: {'preference_score': 0.0, 'patterns': [], 'last_updated': None})
        
        # Session metadata for comprehensive tracking
        self.session_metadata = {
            'session_start': datetime.datetime.now(),
            'total_queries': 0,
            'unique_topics': set(),
            'conversation_depth': 0,
            'user_engagement_score': 0.0,
            'dominant_topics': [],
            'query_patterns': Counter(),
            'response_satisfaction': []
        }
        
        # Fuzzy matching parameters (tuned for optimal performance)
        self.fuzzy_threshold = 0.6  # Lowered from 0.8 to 0.6 for more reasonable matching
        self.semantic_keywords = self._initialize_semantic_keywords()
        
        # Pre-compiled patterns for efficient matching
        self.followup_patterns = [
            r'\b(also|and|what about|how about|additionally|furthermore|moreover)\b',
            r'\b(tell me more|explain further|elaborate|continue)\b',
            r'\b(that|this|it)\b.*\b(sounds|seems|looks|appears)\b',
            r'\b(why|how|when|where|what)\s+(is|are|was|were)\s+(that|this|it)\b'
        ]
        
        self.topic_extraction_patterns = [
            r'\b(machine learning|artificial intelligence|data science|programming|algorithm)\b',
            r'\b(python|java|javascript|c\+\+|sql|html|css)\b',
            r'\b(database|server|network|security|cloud|api)\b',
            r'\b(frontend|backend|fullstack|devops|testing|deployment)\b'
        ]

    def store_in_cache(self, query: str, response: str) -> None:
        """
        Caches the query and its associated response for future lookups.
        This implements the core memoization strategy of dynamic programming.
        
        TODO: IMPLEMENT MEMOIZATION STORAGE
        Steps to implement:
        1. Normalize the query to create a consistent cache key
        2. Create enhanced cache entry with metadata
        3. Update session metadata and tracking structures
        4. Add interaction to conversation context
        5. Handle any errors gracefully
        
        Args:
            query (str): The user's original question.
            response (str): The chatbot's generated reply.
        """
        try:
            # TODO: STEP 1 - Create cache key
            # Use _normalize_query() to create a consistent key for the cache
            # cache_key = self._normalize_query(query)
            
            # TODO: STEP 2 - Create enhanced cache entry
            # Create a dictionary with the following structure:
            # cache_entry = {
            #     'response': response,
            #     'timestamp': datetime.datetime.now(),
            #     'access_count': 1,
            #     'original_query': query,
            #     'query_length': len(query.split()),
            #     'response_length': len(response.split()),
            #     'topics': self._extract_topics(query),
            #     'query_type': self._classify_query_type(query),
            #     'context_relevance': self._calculate_context_relevance(query)
            # }
            
            # TODO: STEP 3 - Store in cache
            # self.response_cache[cache_key] = cache_entry
            
            # TODO: STEP 4 - Update tracking systems
            # Call these methods to update various tracking systems:
            # self._update_session_metadata(query, response)
            # self._update_topic_continuity(query, response)
            # self._update_user_preferences(query, response)
            
            # TODO: STEP 5 - Add to conversation context
            # self._add_to_context(query, response)

        except Exception as e:
            print(f"Error storing in cache: {e}")

    def retrieve_from_cache(self, query: str) -> Optional[str]:
        """
        Retrieves cached responses using dynamic programming principles.
        First tries exact match (O(1) lookup), then fuzzy matching if needed.
        
        TODO: IMPLEMENT CACHE RETRIEVAL ALGORITHM
        Steps to implement:
        1. Normalize query and check for exact match
        2. Update cache metadata if found
        3. If no exact match, perform fuzzy search
        4. Return best match or None
        
        Args:
            query (str): The user's current query.

        Returns:
            Optional[str]: The cached response if available; otherwise, None.
        """
        try:
            # TODO: STEP 1 - Check for exact match
            # cache_key = self._normalize_query(query)
            # if cache_key in self.response_cache:
            #     # Update access metadata
            #     cache_entry = self.response_cache[cache_key]
            #     cache_entry['access_count'] += 1
            #     cache_entry['last_accessed'] = datetime.datetime.now()
            #     cache_entry['context_relevance'] = self._calculate_context_relevance(query)
            #     return cache_entry['response']
            
            # TODO: STEP 2 - Perform fuzzy search if no exact match
            # return self._fuzzy_cache_lookup(query)

        except Exception as e:
            print(f"Error retrieving from cache: {e}")
            return None

    def has_context(self, query: str) -> bool:
        """
        Checks whether the current query has context (exists in cache or similar).
        Uses dynamic programming to efficiently check both exact and fuzzy matches.
        
        TODO: IMPLEMENT CONTEXT DETECTION
        Steps to implement:
        1. Check exact match first (fastest)
        2. Check fuzzy match with lower threshold
        3. Return boolean result
        
        Args:
            query (str): The incoming user question.

        Returns:
            bool: True if the query exists in cache or has high similarity; otherwise, False.
        """
        try:
            # TODO: STEP 1 - Check exact match
            # cache_key = self._normalize_query(query)
            # if cache_key in self.response_cache:
            #     return True
            
            # TODO: STEP 2 - Check fuzzy match with lower threshold
            # fuzzy_result = self._fuzzy_cache_lookup(query, threshold=0.4)
            # return fuzzy_result is not None
            
        except Exception as e:
            print(f"Error checking context: {e}")
            return False

    def _normalize_query(self, query: str) -> str:
        """
        Normalizes queries to create consistent cache keys for dynamic programming.
        This ensures that similar queries map to the same cache entry.
        
        TODO: IMPLEMENT QUERY NORMALIZATION
        Steps to implement:
        1. Convert to lowercase and strip whitespace
        2. Remove punctuation
        3. Handle contractions
        4. Remove extra whitespace
        5. Return normalized string
        
        Args:
            query (str): The original query string.

        Returns:
            str: The normalized query string.
        """
        # TODO: STEP 1 - Basic normalization
        # normalized = query.lower().strip()
        
        # TODO: STEP 2 - Remove punctuation
        # normalized = re.sub(r'[?!.,;:]', '', normalized)
        
        # TODO: STEP 3 - Handle contractions
        # Use the provided contractions dictionary to expand contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "wouldn't": "would not", "shouldn't": "should not",
            "couldn't": "could not", "mustn't": "must not"
        }
        
        # TODO: Replace each contraction with its expansion
        # for contraction, expansion in contractions.items():
        #     normalized = normalized.replace(contraction, expansion)
        
        # TODO: STEP 4 - Remove extra whitespace
        # normalized = re.sub(r'\s+', ' ', normalized)
        
        # TODO: STEP 5 - Return normalized query
        # return normalized

    def _add_to_context(self, query: str, response: str) -> None:
        """
        Implements sliding window technique to manage conversation context.
        This is a key dynamic programming optimization for memory management.
        
        TODO: IMPLEMENT SLIDING WINDOW CONTEXT MANAGEMENT
        Steps to implement:
        1. Create timestamped entries for user and bot
        2. Add entries to conversation context
        3. Implement sliding window pruning
        4. Handle errors gracefully
        
        Args:
            query (str): The user question.
            response (str): The bot's reply.
        """
        try:
            # TODO: STEP 1 - Create timestamped entries
            timestamp = datetime.datetime.now()
            
            # Create user entry with metadata
            # user_entry = {
            #     'type': 'user',
            #     'content': query,
            #     'timestamp': timestamp,
            #     'word_count': len(query.split()),
            #     'topics': self._extract_topics(query),
            #     'query_type': self._classify_query_type(query),
            #     'context_position': len(self.conversation_context)
            # }
            
            # Create bot entry with metadata
            # bot_entry = {
            #     'type': 'bot',
            #     'content': response,
            #     'timestamp': timestamp,
            #     'word_count': len(response.split()),
            #     'topics': self._extract_topics(response),
            #     'response_type': self._classify_response_type(response),
            #     'context_position': len(self.conversation_context) + 1
            # }

            # TODO: STEP 2 - Add to conversation context
            # self.conversation_context.append(user_entry)
            # self.conversation_context.append(bot_entry)

            # TODO: STEP 3 - Implement sliding window pruning
            # if len(self.conversation_context) > self.context_window * 2:
            #     self.conversation_context = self._smart_context_pruning()

        except Exception as e:
            print(f"Error adding to context: {e}")

    def _fuzzy_cache_lookup(self, query: str, threshold: float = None) -> Optional[str]:
        """
        Implements fuzzy matching algorithm for finding similar cached queries.
        This uses dynamic programming principles to optimize similarity calculations.
        
        TODO: IMPLEMENT FUZZY MATCHING ALGORITHM
        Steps to implement:
        1. Set up similarity scoring variables
        2. Calculate multiple similarity metrics for each cache entry
        3. Combine scores using weighted formula
        4. Return best match above threshold
        
        Args:
            query (str): The user's input that didn't yield an exact cache match.
            threshold (float): Custom threshold for matching (uses default if None).

        Returns:
            Optional[str]: A similar cached response, or None if no match found.
        """
        try:
            # TODO: STEP 1 - Initialize variables
            # if threshold is None:
            #     threshold = self.fuzzy_threshold
            # 
            # query_words = set(query.lower().split())
            # best_match = None
            # best_score = 0.0
            
            # TODO: STEP 2 - Loop through cache entries
            # for cache_key, cache_entry in self.response_cache.items():
            #     original_cache_query = cache_entry['original_query']
            #     cache_words = set(original_cache_query.lower().split())
            #     
            #     if query_words and cache_words:
            #         # TODO: Calculate word overlap score (Jaccard similarity)
            #         # overlap = len(query_words.intersection(cache_words))
            #         # union_size = len(query_words.union(cache_words))
            #         # word_score = overlap / union_size if union_size > 0 else 0
            #         
            #         # TODO: Add overlap bonuses
            #         # if overlap >= 2:  # At least 2 words match
            #         #     word_score += 0.5
            #         # elif overlap >= 1:  # At least 1 word matches
            #         #     word_score += 0.3
            #         
            #         # TODO: Calculate semantic similarity
            #         # semantic_score = self._calculate_semantic_similarity(query, original_cache_query)
            #         
            #         # TODO: Get context and frequency scores
            #         # context_score = cache_entry.get('context_relevance', 0.0)
            #         # frequency_score = min(cache_entry['access_count'] / 10.0, 0.1)
            #         
            #         # TODO: Combine scores with weights
            #         # combined_score = (
            #         #     word_score * 0.75 +
            #         #     semantic_score * 0.15 +
            #         #     context_score * 0.05 +
            #         #     frequency_score * 0.05
            #         # )
            #         
            #         # TODO: Update best match if score is better and above threshold
            #         # if combined_score > best_score and combined_score > threshold:
            #         #     best_score = combined_score
            #         #     best_match = cache_entry['response']

            # TODO: STEP 3 - Return best match
            # return best_match

        except Exception as e:
            print(f"Error in fuzzy cache lookup: {e}")
            return None

    def _analyze_context(self, query: str) -> Dict[str, Any]:
        """
        Analyzes query context using dynamic programming to build comprehensive metadata.
        
        TODO: IMPLEMENT CONTEXT ANALYSIS ALGORITHM
        Steps to implement:
        1. Initialize context analysis structure
        2. Detect follow-up patterns using regex
        3. Extract topics from recent conversation
        4. Calculate continuity and engagement scores
        5. Return comprehensive analysis
        
        Args:
            query (str): The user's input.

        Returns:
            Dict[str, Any]: Enhanced metadata including flags and inferred topics.
        """
        try:
            # TODO: STEP 1 - Initialize context analysis structure
            context_info = {
                'is_followup': False,
                'related_topics': [],
                'sentiment_shift': 'neutral',
                'conversation_depth': self.session_metadata['conversation_depth'],
                'topic_continuity_score': 0.0,
                'user_engagement_level': 'medium',
                'query_complexity': 'simple',
                'expected_response_type': 'informational'
            }

            # TODO: STEP 2 - Detect follow-up patterns
            # query_lower = query.lower()
            # for pattern in self.followup_patterns:
            #     if re.search(pattern, query_lower):
            #         context_info['is_followup'] = True
            #         break

            # TODO: STEP 3 - Extract topics from recent conversation
            # recent_queries = [entry['content'] for entry in self.conversation_context[-6:]
            #                   if entry['type'] == 'user']
            # 
            # all_topics = set()
            # for recent_query in recent_queries:
            #     topics = self._extract_topics(recent_query)
            #     all_topics.update(topics)
            #     
            #     # Extract meaningful words
            #     words = recent_query.lower().split()
            #     for word in words:
            #         if len(word) > 3 and word not in stop_words:
            #             all_topics.add(word)
            # 
            # context_info['related_topics'] = list(all_topics)

            # TODO: STEP 4 - Calculate continuity and engagement
            # if all_topics:
            #     continuity_scores = [self.topic_continuity[topic]['score'] for topic in all_topics]
            #     context_info['topic_continuity_score'] = sum(continuity_scores) / len(continuity_scores)
            # 
            # context_info['query_complexity'] = self._analyze_query_complexity(query)
            # context_info['user_engagement_level'] = self._calculate_user_engagement()
            # context_info['expected_response_type'] = self._predict_response_type(query)

            return context_info

        except Exception as e:
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

    # HELPER METHODS WITH HARDCODED VALUES (No implementation needed)

    def _initialize_semantic_keywords(self) -> Dict[str, List[str]]:
        """Initialize semantic keyword groups for better matching."""
        return {
            'programming': ['code', 'coding', 'program', 'software', 'development', 'algorithm', 'programming', 'python', 'java', 'language', 'methods', 'basics'],
            'data': ['information', 'dataset', 'statistics', 'analysis', 'mining', 'data', 'structures'],
            'learning': ['study', 'education', 'tutorial', 'guide', 'teach', 'explain', 'learning', 'machine', 'neural', 'networks'],
            'technology': ['tech', 'digital', 'computer', 'system', 'network', 'platform'],
            'web': ['website', 'internet', 'online', 'browser', 'html', 'css', 'javascript']
        }

    def _extract_topics(self, text: str) -> List[str]:
        """
        TODO: IMPLEMENT TOPIC EXTRACTION
        Extract topics using pattern matching and keyword analysis.
        
        Steps:
        1. Use topic_extraction_patterns to find pattern-based topics
        2. Extract meaningful words (length > 3, not in stop words)
        3. Extract compound phrases
        4. Return unique topics list
        """
        # TODO: Implement topic extraction logic
        return []  # Placeholder

    def _classify_query_type(self, query: str) -> str:
        """
        TODO: IMPLEMENT QUERY CLASSIFICATION
        Classify queries into types: definition, procedural, causal, comparison, etc.
        
        Use keyword detection to determine query type.
        """
        # TODO: Implement query type classification
        return 'question'  # Placeholder

    def _classify_response_type(self, response: str) -> str:
        """
        TODO: IMPLEMENT RESPONSE CLASSIFICATION
        Classify responses as: brief, procedural, example, detailed, informational.
        """
        # TODO: Implement response type classification
        return 'informational'  # Placeholder

    def _calculate_context_relevance(self, query: str) -> float:
        """
        TODO: IMPLEMENT CONTEXT RELEVANCE CALCULATION
        Calculate how relevant a query is to current conversation using topic overlap.
        """
        # TODO: Implement context relevance calculation
        return 0.0  # Placeholder

    def _calculate_semantic_similarity(self, query1: str, query2: str) -> float:
        """
        TODO: IMPLEMENT SEMANTIC SIMILARITY CALCULATION
        Calculate semantic similarity using word overlap, semantic categories, and topics.
        
        Steps:
        1. Calculate word similarity (Jaccard index)
        2. Check semantic category overlap
        3. Calculate topic similarity
        4. Combine with weighted formula
        """
        # TODO: Implement semantic similarity calculation
        return 0.0  # Placeholder

    def _update_session_metadata(self, query: str, response: str) -> None:
        """
        TODO: IMPLEMENT SESSION METADATA UPDATES
        Update session-level tracking including query count, topics, patterns.
        """
        # TODO: Implement session metadata updates
        pass

    def _update_topic_continuity(self, query: str, response: str) -> None:
        """
        TODO: IMPLEMENT TOPIC CONTINUITY TRACKING
        Update topic frequency and scores based on new interactions.
        """
        # TODO: Implement topic continuity updates
        pass

    def _update_user_preferences(self, query: str, response: str) -> None:
        """
        TODO: IMPLEMENT USER PREFERENCE LEARNING
        Learn user preferences from interaction patterns.
        """
        # TODO: Implement user preference updates
        pass

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze the complexity level of a query based on word count."""
        word_count = len(query.split())
        
        if word_count <= 3:
            return 'simple'
        elif word_count <= 8:
            return 'medium'
        else:
            return 'complex'

    def _calculate_user_engagement(self) -> str:
        """Calculate user engagement level based on conversation patterns."""
        if self.session_metadata['total_queries'] == 0:
            return 'low'
        
        # TODO: IMPLEMENT ENGAGEMENT CALCULATION
        # Calculate based on average query length and interaction patterns
        return 'medium'  # Placeholder

    def _predict_response_type(self, query: str) -> str:
        """Predict expected response type based on query analysis."""
        query_type = self._classify_query_type(query)
        
        type_mapping = {
            'definition': 'explanatory',
            'procedural': 'step-by-step',
            'causal': 'analytical',
            'comparison': 'comparative',
            'example': 'illustrative',
            'question': 'informational',
            'statement': 'confirmatory'
        }
        
        return type_mapping.get(query_type, 'informational')

    def _smart_context_pruning(self) -> List[Dict[str, Any]]:
        """
        TODO: IMPLEMENT SMART CONTEXT PRUNING
        Intelligently prune context while preserving important information.
        
        Steps:
        1. Keep most recent entries
        2. Preserve entries with high topic continuity scores
        3. Sort by timestamp and limit to context window
        """
        # TODO: Implement smart pruning algorithm
        return self.conversation_context[-self.context_window * 2:]  # Simple fallback

    # UTILITY METHODS (Already implemented - no changes needed)

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the current conversation session."""
        return {
            'session_duration': datetime.datetime.now() - self.session_metadata['session_start'],
            'total_interactions': self.session_metadata['total_queries'],
            'dominant_topics': self.session_metadata['dominant_topics'],
            'user_engagement': self._calculate_user_engagement(),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'conversation_depth': self.session_metadata['conversation_depth'],
            'topic_diversity': len(self.session_metadata['unique_topics']),
            'most_common_query_types': self.session_metadata['query_patterns'].most_common(3)
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate the cache hit rate for this session."""
        if self.session_metadata['total_queries'] == 0:
            return 0.0
        
        total_accesses = sum(entry['access_count'] for entry in self.response_cache.values())
        unique_queries = len(self.response_cache)
        
        if unique_queries == 0:
            return 0.0
            
        return min(total_accesses / self.session_metadata['total_queries'], 1.0)

    def get_topic_insights(self) -> Dict[str, Any]:
        """Get insights about topic patterns and continuity."""
        return {
            'active_topics': dict(self.topic_continuity),
            'topic_trends': self._analyze_topic_trends(),
            'conversation_flow': self._analyze_conversation_flow()
        }

    def _analyze_topic_trends(self) -> Dict[str, str]:
        """Analyze trending topics in the conversation."""
        trends = {}
        for topic, data in self.topic_continuity.items():
            if data['frequency'] > 2:
                trends[topic] = 'trending'
            elif data['frequency'] > 1:
                trends[topic] = 'active'
            else:
                trends[topic] = 'emerging'
        return trends

    def _analyze_conversation_flow(self) -> List[str]:
        """Analyze the flow of conversation topics."""
        flow = []
        for entry in self.conversation_context:
            if entry['type'] == 'user' and 'topics' in entry:
                flow.extend(entry['topics'])
        return flow[-10:]  # Return last 10 topics in flow