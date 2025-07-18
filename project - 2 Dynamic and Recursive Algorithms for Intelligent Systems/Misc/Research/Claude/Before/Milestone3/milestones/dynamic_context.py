"""
Milestone 2: Integrate Dynamic Programming for Context-Aware Responses

This module implements dynamic context tracking to enhance the chatbot's ability
to remember previous interactions and provide more coherent, contextual replies.
It simulates dynamic programming by caching previous results and maintaining
a sliding window of recent conversations.
"""

import datetime
from typing import Dict, Optional, Any, List


class DynamicContext:
    """
    This class manages conversation context using memoization (a dynamic programming strategy)
    and a contextual history buffer. It allows the chatbot to:
    - Recall previous questions and answers
    - Detect follow-up questions
    - Infer conversation topics and user preferences
    """

    def __init__(self):
        """
        Initializes all necessary data structures:
        - response_cache: stores normalized queries and their answers
        - conversation_context: stores a timeline of user/bot exchanges
        - context_window: how many past interactions are remembered
        - topic_continuity: placeholder for topic tracking (not fully used)
        - user_preferences: placeholder for learning long-term preferences
        """
        self.response_cache = {}  # Dictionary to store cached query/response pairs
        self.conversation_context = []  # List of user-bot message pairs (chronological)
        self.context_window = 10  # Only store the latest 10 question-answer turns
        self.topic_continuity = {}  # Reserved for future topic-tracking features
        self.user_preferences = {}  # Reserved for future preference-tracking features

    def store_in_cache(self, query: str, response: str) -> None:
        """
        Caches the query and its associated response for future lookups.
        Also adds both the query and response to the context history.

        Args:
            query (str): The user’s original question.
            response (str): The chatbot’s generated reply.
        """
        try:
            cache_key = self._normalize_query(query)  # Normalize for consistent matching
            self.response_cache[cache_key] = {
                'response': response,
                'timestamp': datetime.datetime.now(),
                'access_count': 1,
                'original_query': query
            }

            # Also push this interaction into the rolling context window
            self._add_to_context(query, response)

        except Exception as e:
            print(f"Error storing in cache: {e}")

    def retrieve_from_cache(self, query: str) -> Optional[str]:
        """
        Attempts to retrieve an answer from the cache using the exact normalized query.
        If not found, it performs a fuzzy search based on word overlap.

        Args:
            query (str): The user’s current query.

        Returns:
            Optional[str]: The cached response if available; otherwise, None.
        """
        try:
            cache_key = self._normalize_query(query)

            if cache_key in self.response_cache:
                cache_entry = self.response_cache[cache_key]
                cache_entry['access_count'] += 1
                cache_entry['last_accessed'] = datetime.datetime.now()
                return cache_entry['response']

            # Try approximate match if no exact match
            return self._fuzzy_cache_lookup(query)

        except Exception as e:
            print(f"Error retrieving from cache: {e}")
            return None

    def has_context(self, query: str) -> bool:
        """
        Checks whether the current query already exists in the cached responses.

        Args:
            query (str): The incoming user question.

        Returns:
            bool: True if the query exists in cache; otherwise, False.
        """
        try:
            cache_key = self._normalize_query(query)
            return cache_key in self.response_cache
        except Exception as e:
            print(f"Error checking context: {e}")
            return False

    def _normalize_query(self, query: str) -> str:
        """
        A helper method to standardize query format before caching.
        This improves hit rate and avoids issues from casing or punctuation.

        Args:
            query (str): The original query string.

        Returns:
            str: The normalized query string.
        """
        return query.lower().strip().replace('?', '').replace('!', '')

    def _add_to_context(self, query: str, response: str) -> None:
        """
        Adds the user query and the bot response as a pair to the
        context history, preserving order and timestamps.

        Args:
            query (str): The user question.
            response (str): The bot’s reply.
        """
        try:
            timestamp = datetime.datetime.now()

            # Save user question
            self.conversation_context.append({
                'type': 'user',
                'content': query,
                'timestamp': timestamp
            })

            # Save bot reply
            self.conversation_context.append({
                'type': 'bot',
                'content': response,
                'timestamp': timestamp
            })

            # Limit history to context_window * 2 (user+bot entries)
            if len(self.conversation_context) > self.context_window * 2:
                self.conversation_context = self.conversation_context[-self.context_window * 2:]

        except Exception as e:
            print(f"Error adding to context: {e}")

    def _fuzzy_cache_lookup(self, query: str) -> Optional[str]:
        """
        Attempts to find a close match to the current query based on word similarity.
        Uses word overlap to determine match strength.

        Args:
            query (str): The user’s input that didn’t yield an exact cache match.

        Returns:
            Optional[str]: A similar cached response, or None if no match found.
        """
        try:
            query_words = set(query.lower().split())
            best_match = None
            best_score = 0.0

            for cache_key, cache_entry in self.response_cache.items():
                cache_words = set(cache_key.split())
                if query_words and cache_words:
                    overlap = len(query_words.intersection(cache_words))
                    score = overlap / len(query_words.union(cache_words))

                    # Only consider it a match if the score is very high (> 0.7)
                    if score > best_score and score > 0.7:
                        best_score = score
                        best_match = cache_entry['response']

            return best_match

        except Exception as e:
            print(f"Error in fuzzy cache lookup: {e}")
            return None

    def _analyze_context(self, query: str) -> Dict[str, Any]:
        """
        Analyze the given query in the context of recent conversation history.
        This is useful for detecting:
        - Whether the query is a follow-up
        - Related past topics
        - Potential sentiment shift

        Args:
            query (str): The user’s input.

        Returns:
            Dict[str, Any]: Metadata including flags and inferred topics.
        """
        try:
            context_info = {
                'is_followup': False,
                'related_topics': [],
                'sentiment_shift': 'neutral'
            }

            # Detect possible follow-up language
            followup_words = ['also', 'and', 'what about', 'how about', 'additionally']
            if any(word in query.lower() for word in followup_words):
                context_info['is_followup'] = True

            # Look back through the last 3 user queries
            recent_queries = [entry['content'] for entry in self.conversation_context[-6:]
                              if entry['type'] == 'user']

            for recent_query in recent_queries:
                words = recent_query.lower().split()
                important_words = [w for w in words if len(w) > 4 and w not in ['what', 'how', 'where', 'when', 'why']]
                context_info['related_topics'].extend(important_words)

            # Remove duplicates
            context_info['related_topics'] = list(set(context_info['related_topics']))

            return context_info

        except Exception as e:
            print(f"Error analyzing context: {e}")
            return {
                'is_followup': False,
                'related_topics': [],
                'sentiment_shift': 'neutral'
            }
