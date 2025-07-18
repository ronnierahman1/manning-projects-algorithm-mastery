"""
Knowledge Base Module

This module manages the chatbot's QA knowledge base.
- It loads SQuAD-style JSON data from file.
- Supports exact and fuzzy search with unified fuzzy matching logic.
- Provides fallback sample data when JSON loading fails.

Milestone: Implemented fully in Milestone 4.
Enhanced: Added advanced search, analytics, caching, and persistence features.
Refactored: Unified fuzzy matching and text processing utilities.
"""

import json
import os
import difflib
import string
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import pickle
import hashlib


class KnowledgeBase:
    """
    Class to manage the knowledge base for AI chatbot.
    - Loads question/answer/context triplets into memory
    - Performs exact and fuzzy search for user queries
    - Enhanced with analytics, caching, and advanced search capabilities
    - Unified fuzzy matching and text processing utilities
    """

    # Class-level constants for fuzzy matching
    DEFAULT_FUZZY_THRESHOLDS = [0.9, 0.8, 0.75, 0.7, 0.6, 0.5]
    PUNCTUATION_PATTERN = re.compile(r'[.?!]+$')

    def __init__(self, data_path: str):
        """
        Initialize the knowledge base and load data.

        Args:
            data_path (str): Path to the dev-v2.0.json file
        """
        self.qa_pairs = []        # Cached list of all parsed QA entries
        self.data_path = data_path
        
        # Enhanced features
        self.search_analytics = defaultdict(int)  # Track search patterns
        self.query_cache = {}     # Cache for frequently asked questions
        self.synonyms = {}        # Synonym dictionary for better matching
        self.categories = defaultdict(list)  # Categorized QA pairs
        self.search_history = []  # Track search history
        self.response_times = []  # Track performance metrics
        self.feedback_data = {}   # Store user feedback on answers
        self.custom_qa_pairs = [] # User-added QA pairs
        self.index_cache = {}     # Cached search indices
        
        self.load_data(data_path)
        self._build_search_index()
        self._load_enhancements()

    # --- Text Processing Utilities ---

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by stripping, lowercasing, and removing end punctuation.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        return KnowledgeBase.PUNCTUATION_PATTERN.sub('', text.strip().lower()).strip()

    @staticmethod
    def clean_words(text: str) -> set:
        """
        Clean and extract words from text, removing punctuation.
        
        Args:
            text (str): Text to process
            
        Returns:
            set: Set of cleaned words
        """
        if not text:
            return set()
            
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove punctuation from each word
        cleaned_words = set()
        for word in words:
            # Remove all punctuation from the word
            clean_word = ''.join(char for char in word if char not in string.punctuation)
            if clean_word:  # Only add non-empty words
                cleaned_words.add(clean_word)
        
        return cleaned_words

    # --- Unified Fuzzy Matching ---

    def fuzzy_match(self, query: str, threshold: float = 0.6, max_results: int = 1) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform fuzzy matching against the knowledge base.
        
        Args:
            query (str): Query string to match
            threshold (float): Minimum similarity threshold (0.0 to 1.0)
            max_results (int): Maximum number of results to return
            
        Returns:
            List of tuples containing (qa_dict, similarity_score) sorted by score
        """
        try:
            if not query or not query.strip():
                return []
                
            query_normalized = self.normalize_text(query)
            if not query_normalized:
                return []
                
            matches = []
            
            # Search through all QA pairs
            all_qa_pairs = getattr(self, 'qa_pairs', []) + getattr(self, 'custom_qa_pairs', [])
            
            for qa in all_qa_pairs:
                if not qa or 'question' not in qa:
                    continue
                    
                question_normalized = self.normalize_text(qa['question'])
                
                # Calculate similarity score
                import difflib
                similarity = difflib.SequenceMatcher(None, query_normalized, question_normalized).ratio()
                
                if similarity >= threshold:
                    matches.append((qa, similarity))
            
            # Sort by similarity score (highest first) and limit results
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:max_results]
            
        except Exception as e:
            print(f"Error in fuzzy_match: {e}")
            return []  # Always return empty list on error
        
    def fuzzy_match_with_thresholds(self, query: str, thresholds: List[float] = None) -> Optional[Tuple[str, float]]:
        """
        Perform fuzzy matching with multiple thresholds, returning the first good match.
        
        Args:
            query (str): Query string to match
            thresholds (List[float]): List of thresholds to try (highest to lowest)
            
        Returns:
            Tuple of (answer, similarity_score) or None if no match found
        """
        if thresholds is None:
            thresholds = self.DEFAULT_FUZZY_THRESHOLDS
            
        try:
            for threshold in thresholds:
                matches = self.fuzzy_match(query, threshold, max_results=1)
                if matches and len(matches) > 0:
                    qa_dict, similarity = matches[0]
                    # Update usage statistics
                    self._update_usage_stats(qa_dict)
                    answer = qa_dict.get('answer', '')
                    # Ensure we return a proper tuple with 2 elements
                    return (answer, similarity)
                    
            # Explicitly return None if no matches found
            return None
            
        except Exception as e:
            print(f"Error in fuzzy_match_with_thresholds: {e}")
            # Always return None on error, never an empty tuple
            return None
    
    # --- Main Query Processing ---

    def load_data(self, data_path: str) -> None:
        """
        Parses the SQuAD-style JSON dataset and populates qa_pairs in memory.

        Fallbacks to default questions if file is invalid.
        """
        if not os.path.exists(data_path):
            print(f"⚠️ Warning: {data_path} not found.")
            self._create_sample_data()
            return

        try:
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            if "data" not in data:
                raise ValueError("JSON format is invalid. Missing 'data' key.")

            for article in data["data"]:
                for paragraph in article.get("paragraphs", []):
                    context = paragraph.get("context", "")
                    for qa in paragraph.get("qas", []):
                        question = qa.get("question", "").strip()
                        if not question:
                            continue

                        answers = qa.get("answers", [])
                        if answers:
                            answer = answers[0].get("text", "").strip()
                        else:
                            answer = f"Based on the context: {context[:200]}..." if context else "Answer not available."

                        if question and answer:
                            qa_entry = {
                                "question": question,
                                "answer": answer,
                                "context": context,
                                "id": len(self.qa_pairs),
                                "usage_count": 0,
                                "last_used": None,
                                "confidence": 1.0
                            }
                            self.qa_pairs.append(qa_entry)

            print(f"✅ Loaded {len(self.qa_pairs)} QA pairs from knowledge base.")

        except Exception as e:
            print(f"❌ Failed to load knowledge base: {e}")
            self._create_sample_data()

    def _create_sample_data(self) -> None:
        """
        Creates fallback hardcoded data if JSON file is not usable.
        """
        self.qa_pairs = [
            {
                'question': 'What is AI?',
                'answer': 'AI stands for artificial intelligence.',
                'context': 'AI basics',
                'id': 0,
                'usage_count': 0,
                'last_used': None,
                'confidence': 1.0
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
        Returns number of loaded QA pairs.
        """
        return len(self.qa_pairs) + len(self.custom_qa_pairs)

    def get_exact_match_answer(self, query: str) -> str:
        """
        Main query processor.
        1. Performs exact match after stripping end punctuation.
        2. Falls back to fuzzy match at decreasing similarity thresholds.
        """
        start_time = time.time()
        
        try:
            if not query or not query.strip():
                return "Please enter a question."

            # Check cache first
            query_hash = self._hash_query(query)
            if query_hash in self.query_cache:
                self._update_analytics(query, True)
                return self.query_cache[query_hash]

            query_normalized = self.normalize_text(query)

            # --- 1. Exact Match ---
            all_qa_pairs = getattr(self, 'qa_pairs', []) + getattr(self, 'custom_qa_pairs', [])
            
            for qa in all_qa_pairs:
                if not qa or 'question' not in qa:
                    continue
                    
                question_normalized = self.normalize_text(qa['question'])
                if query_normalized == question_normalized:
                    answer = qa.get('answer', '')
                    self._update_usage_stats(qa)
                    self._cache_result(query_hash, answer)
                    self._update_analytics(query, True)
                    return answer

            # No exact match found
            return f"I couldn't find a good match for: '{query}'. Please rephrase."

        except Exception as e:
            print(f"❌ Error in get_exact_match_answer: {e}")
            return "An error occurred while processing your question."
        
        finally:
            response_time = time.time() - start_time
            if hasattr(self, 'response_times'):
                self.response_times.append(response_time)
            if hasattr(self, 'search_history'):
                self.search_history.append({
                    'query': query,
                    'timestamp': datetime.now(),
                    'response_time': response_time
                })

    # --- Enhanced Methods ---

    def search_by_keyword(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for QA pairs containing a specific keyword.
        
        Args:
            keyword (str): Keyword to search for
            limit (int): Maximum number of results to return
            
        Returns:
            List of matching QA pairs with relevance scores
        """
        keyword_lower = keyword.lower()
        results = []
        
        for qa in self.qa_pairs + self.custom_qa_pairs:
            score = 0
            
            # Check question
            if keyword_lower in qa['question'].lower():
                score += 3
            
            # Check answer
            if keyword_lower in qa['answer'].lower():
                score += 2
                
            # Check context
            if keyword_lower in qa.get('context', '').lower():
                score += 1
                
            if score > 0:
                result = qa.copy()
                result['relevance_score'] = score
                results.append(result)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:limit]

    def get_similar_questions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find questions similar to the given query.
        
        Args:
            query (str): Query to find similar questions for
            limit (int): Maximum number of similar questions to return
            
        Returns:
            List of similar questions with similarity scores
        """
        matches = self.fuzzy_match(query, threshold=0.3, max_results=limit)
        
        similarities = []
        for qa, similarity in matches:
            similarities.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'similarity': similarity,
                'usage_count': qa.get('usage_count', 0)
            })
        
        return similarities

    def add_custom_qa(self, question: str, answer: str, context: str = "") -> bool:
        """
        Add a custom QA pair to the knowledge base.
        
        Args:
            question (str): Question text
            answer (str): Answer text
            context (str): Optional context
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            if not question.strip() or not answer.strip():
                return False
                
            qa_entry = {
                'question': question.strip(),
                'answer': answer.strip(),
                'context': context.strip(),
                'id': len(self.qa_pairs) + len(self.custom_qa_pairs),
                'usage_count': 0,
                'last_used': None,
                'confidence': 1.0,
                'custom': True
            }
            
            self.custom_qa_pairs.append(qa_entry)
            self._save_custom_data()
            return True
            
        except Exception as e:
            print(f"❌ Error adding custom QA: {e}")
            return False

    def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about knowledge base usage.
        
        Returns:
            Dictionary containing various analytics metrics
        """
        total_queries = sum(self.search_analytics.values())
        successful_queries = self.search_analytics.get('successful', 0)
        success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        # Most popular questions
        popular_questions = sorted(
            [(qa['question'], qa.get('usage_count', 0)) 
             for qa in self.qa_pairs + self.custom_qa_pairs],
            key=lambda x: x[1], reverse=True
        )[:10]
        
        # Average response time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
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
        Get recent search history.
        
        Args:
            limit (int): Maximum number of recent searches to return
            
        Returns:
            List of recent search entries
        """
        return self.search_history[-limit:]

    def provide_feedback(self, query: str, helpful: bool, comment: str = "") -> None:
        """
        Collect user feedback on answers.
        
        Args:
            query (str): The original query
            helpful (bool): Whether the answer was helpful
            comment (str): Optional comment
        """
        query_hash = self._hash_query(query)
        self.feedback_data[query_hash] = {
            'helpful': helpful,
            'comment': comment,
            'timestamp': datetime.now()
        }
        self._save_feedback_data()

    def export_knowledge_base(self, filepath: str) -> bool:
        """
        Export the knowledge base to a JSON file.
        
        Args:
            filepath (str): Path to save the exported data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            export_data = {
                'qa_pairs': self.qa_pairs,
                'custom_qa_pairs': self.custom_qa_pairs,
                'analytics': self.get_analytics(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting knowledge base: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        print("✅ Query cache cleared.")

    def optimize_knowledge_base(self) -> Dict[str, Any]:
        """
        Optimize the knowledge base by removing duplicates and low-usage entries.
        
        Returns:
            Dictionary with optimization results
        """
        original_size = len(self.qa_pairs)
        
        # Remove duplicates
        seen_questions = set()
        unique_qa_pairs = []
        
        for qa in self.qa_pairs:
            q_normalized = self.normalize_text(qa['question'])
            if q_normalized not in seen_questions:
                seen_questions.add(q_normalized)
                unique_qa_pairs.append(qa)
        
        duplicates_removed = original_size - len(unique_qa_pairs)
        self.qa_pairs = unique_qa_pairs
        
        # Rebuild search index
        self._build_search_index()
        
        return {
            'original_size': original_size,
            'optimized_size': len(self.qa_pairs),
            'duplicates_removed': duplicates_removed,
            'space_saved': f"{duplicates_removed / original_size * 100:.1f}%" if original_size > 0 else "0%"
        }

    # --- Private Helper Methods ---

    def _build_search_index(self) -> None:
        """Build search indices for faster lookups."""
        self.index_cache = {
            'questions': [qa['question'].lower() for qa in self.qa_pairs],
            'answers': [qa['answer'].lower() for qa in self.qa_pairs],
            'contexts': [qa.get('context', '').lower() for qa in self.qa_pairs]
        }

    def _load_enhancements(self) -> None:
        """Load enhancement data from disk."""
        try:
            # Load custom QA pairs
            if os.path.exists('custom_qa.json'):
                with open('custom_qa.json', 'r', encoding='utf-8') as f:
                    self.custom_qa_pairs = json.load(f)
            
            # Load feedback data
            if os.path.exists('feedback.json'):
                with open('feedback.json', 'r', encoding='utf-8') as f:
                    feedback_raw = json.load(f)
                    # Convert string timestamps back to datetime objects
                    for key, value in feedback_raw.items():
                        if 'timestamp' in value:
                            value['timestamp'] = datetime.fromisoformat(value['timestamp'])
                    self.feedback_data = feedback_raw
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not load enhancement data: {e}")

    def _save_custom_data(self) -> None:
        """Save custom QA pairs to disk."""
        try:
            with open('custom_qa.json', 'w', encoding='utf-8') as f:
                json.dump(self.custom_qa_pairs, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Warning: Could not save custom QA data: {e}")

    def _save_feedback_data(self) -> None:
        """Save feedback data to disk."""
        try:
            # Convert datetime objects to strings for JSON serialization
            feedback_serializable = {}
            for key, value in self.feedback_data.items():
                feedback_serializable[key] = value.copy()
                if 'timestamp' in value:
                    feedback_serializable[key]['timestamp'] = value['timestamp'].isoformat()
            
            with open('feedback.json', 'w', encoding='utf-8') as f:
                json.dump(feedback_serializable, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Warning: Could not save feedback data: {e}")

    def _hash_query(self, query: str) -> str:
        """Create a hash for query caching."""
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def _cache_result(self, query_hash: str, answer: str) -> None:
        """Cache a query result."""
        self.query_cache[query_hash] = answer
        
        # Limit cache size
        if len(self.query_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.query_cache.keys())[:100]
            for key in oldest_keys:
                del self.query_cache[key]

    def _update_analytics(self, query: str, success: bool) -> None:
        """Update search analytics."""
        if success:
            self.search_analytics['successful'] += 1
        else:
            self.search_analytics['failed'] += 1
        
        self.search_analytics['total'] += 1

    def _update_usage_stats(self, qa: Dict[str, Any]) -> None:
        """Update usage statistics for a QA pair."""
        qa['usage_count'] = qa.get('usage_count', 0) + 1
        qa['last_used'] = datetime.now()