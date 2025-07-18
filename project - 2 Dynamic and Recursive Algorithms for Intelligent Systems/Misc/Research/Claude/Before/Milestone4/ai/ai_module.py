"""
AI Module for Enhanced Query Processing

This module is designed to add intelligent behaviors to the chatbot by:
1. Detecting the sentiment of the user query (positive, negative, neutral)
2. Expanding the query using synonyms to improve matching success
3. Performing fuzzy matching between the user query and stored knowledge base questions

These tools together help improve the chatbot’s flexibility, personalization, and robustness.
"""

import re
import difflib
from textblob import TextBlob
from typing import List, Dict, Optional, Tuple


class AIModule:
    """
    Core AI augmentation module to improve chatbot query understanding.
    This class enhances matching performance via:
    - Sentiment detection
    - Query expansion using synonyms
    - Fuzzy matching of queries against known question-answer pairs
    """

    def __init__(self):
        """
        Initialize the AI module by defining a basic synonym dictionary.
        This dictionary is used to substitute certain key words in a query
        with related alternatives to expand possible match opportunities.
        """
        self.synonym_dict = {
            'what': ['which', 'how', 'where', 'when', 'why'],
            'how': ['what', 'which', 'where', 'when', 'why'],
            'where': ['what', 'which', 'how', 'when', 'why'],
            'when': ['what', 'which', 'how', 'where', 'why'],
            'why': ['what', 'which', 'how', 'where', 'when'],
            'who': ['what', 'which', 'whom'],
            'good': ['great', 'excellent', 'wonderful', 'amazing', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'negative'],
            'big': ['large', 'huge', 'enormous', 'massive', 'giant'],
            'small': ['little', 'tiny', 'minute', 'miniature', 'compact'],
            'fast': ['quick', 'rapid', 'speedy', 'swift', 'hasty'],
            'slow': ['sluggish', 'gradual', 'leisurely', 'unhurried']
        }

    def detect_sentiment(self, query: str) -> str:
        """
        Analyze the sentiment of a user query using the TextBlob NLP library.

        Args:
            query (str): User's input sentence or question

        Returns:
            str: 'positive', 'negative', or 'neutral' based on polarity score

        Example:
            Input: "I love this chatbot!"
            Output: 'positive'
        """
        try:
            blob = TextBlob(query)
            polarity = blob.sentiment.polarity

            # Sentiment scale: -1 (very negative) to +1 (very positive)
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'

        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 'neutral'

    def expand_query(self, query: str) -> List[str]:
        """
        Generate alternative versions of a query by replacing known words with synonyms.
        This improves the likelihood of matching the question with the knowledge base.

        Args:
            query (str): Original user query

        Returns:
            List[str]: List of the original and synonym-expanded queries

        Example:
            Input: "What is fast?"
            Output: ['what is fast?', 'which is fast?', 'how is fast?', 'where is fast?', ...]
        """
        expanded_queries = [query.lower()]  # Always include original query
        words = query.lower().split()

        try:
            for word in words:
                if word in self.synonym_dict:
                    for synonym in self.synonym_dict[word]:
                        # Replace the word with its synonym in a copy of the query
                        expanded_query = query.lower().replace(word, synonym)
                        if expanded_query not in expanded_queries:
                            expanded_queries.append(expanded_query)
        except Exception as e:
            print(f"Query expansion error: {e}")

        return expanded_queries

    def fuzzy_match(self, query: str, knowledge_base) -> Optional[Tuple[str, str, float]]:
        """
        Match the given query against the questions in the knowledge base
        using fuzzy logic based on text similarity.

        Args:
            query (str): User’s input sentence
            knowledge_base: An instance containing QA pairs in `.qa_pairs`

        Returns:
            Optional[Tuple[str, str, float]]: Returns a tuple of best matching
            (question, answer, match_score) or None if no good match is found.

        Steps:
        - Loop through all questions in the knowledge base
        - Try exact matching
        - If no exact match, apply:
            1. Sequence similarity
            2. Token overlap
            3. Substring presence
        - Choose the one with the highest similarity above a threshold (0.6)

        Example:
            Input: "Where is the Eiffel Tower?"
            Output: ("Where is Paris?", "Paris is in France.", 0.73)
        """
        try:
            if not hasattr(knowledge_base, 'qa_pairs') or not knowledge_base.qa_pairs:
                return None

            query_lower = query.lower().strip()
            best_match = None
            best_score = 0.0

            for qa_pair in knowledge_base.qa_pairs:
                question = qa_pair.get('question', '').lower().strip()
                if not question:
                    continue

                # If there's an exact string match, return it immediately
                if query_lower == question:
                    return question, qa_pair.get('answer', ''), 1.0

                scores = []

                # Method 1: Character-level sequence similarity (difflib)
                seq_score = difflib.SequenceMatcher(None, query_lower, question).ratio()
                scores.append(seq_score)

                # Method 2: Token overlap (common words / total words)
                query_tokens = set(query_lower.split())
                question_tokens = set(question.split())
                if query_tokens and question_tokens:
                    overlap = len(query_tokens.intersection(question_tokens))
                    token_score = overlap / max(len(query_tokens), len(question_tokens))
                    scores.append(token_score)

                # Method 3: Substring presence
                if query_lower in question or question in query_lower:
                    scores.append(0.8)  # Boost score for partial containment

                # Use the maximum score out of all similarity methods
                final_score = max(scores) if scores else 0.0

                if final_score > best_score and final_score > 0.6:
                    best_score = final_score
                    best_match = (question, qa_pair.get('answer', ''), final_score)

            return best_match

        except Exception as e:
            print(f"Fuzzy matching error: {e}")
            return None
