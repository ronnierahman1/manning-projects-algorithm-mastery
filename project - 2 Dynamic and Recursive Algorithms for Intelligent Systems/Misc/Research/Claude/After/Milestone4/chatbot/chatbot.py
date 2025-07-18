import datetime
import random
import re
import string
from typing import List, Tuple, Optional, Dict, Any
from functools import lru_cache

from ai.ai_module import AIModule
from chatbot.knowledge_base import KnowledgeBase

class AIChatbot:
    DEFAULT_NO_MATCH_MESSAGE = "I don't have specific information about that topic. Could you try rephrasing your question?"
    
    # Class-level constants
    PUNCTUATION_PATTERN = re.compile(r'[.?!]+$')
    
    # Pre-compiled patterns for faster matching
    GREETING_WORDS = frozenset(['hello', 'hi', 'hey'])
    GOODBYE_WORDS = frozenset(['bye', 'goodbye'])
    TIME_WORDS = frozenset(['time'])
    DATE_WORDS = frozenset(['date', 'today'])
    HELP_PHRASES = frozenset(['what can you do', 'help me', 'how do you work'])
    IDENTITY_PHRASES = frozenset(['who are you', 'your name'])
    
    def __init__(self, data_path: str):
        try:
            self.knowledge_base = KnowledgeBase(data_path)
            self.ai_module = AIModule()
            self.qa_cache = self.knowledge_base.qa_pairs
            self.conversation_history = []
            
            # Pre-define responses to avoid repeated list creation
            self.default_responses = [
                "I'm here to help! What would you like to know?",
                "That's an interesting question. Could you provide more details?",
                "I'd be happy to assist you with that. Can you elaborate?",
                "Let me help you with that. What specific information are you looking for?"
            ]
            
            self.positive_starters = [
                "Great question!", "I'm happy to help!", "Excellent!", "You're on the right track!",
                "That's a thoughtful query.", "Absolutely!", "Interesting point!", "Let's dive into that.",
                "That's an insightful question!", "You've picked a great topic."
            ]
            
            self.supportive_starters = [
                "I understand this might be concerning.", "Let me help clarify this for you.",
                "You're doing great asking that.", "Don't worry, I've got you covered.",
                "Happy to help!", "You're asking the right person.",
                "Let's work through this together.", "That's what I'm here for.",
                "I'll do my best to guide you.", "You're not alone—I'm here to help."
            ]
            
            print("Chatbot initialized successfully!")
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            raise

    def handle_query(self, query: str) -> Tuple[str, bool, float]:
        """Handle user query with optimized processing."""
        try:
            # Early return for empty queries
            if not query or not query.strip():
                return "Please ask me a question!", False, 0.0

            query = query.strip()
            
            # Add to conversation history
            self._add_to_history('user', query)
            
            # Process the query
            response_text, is_fuzzy, threshold = self._process_query(query)
            
            # Handle empty responses (fallback to generate_response)
            if not response_text or response_text.strip() == "":
                response_text = self.generate_response(query)
                is_fuzzy = False
                threshold = 0.0
            
            # Apply sentiment-based tone (only if needed)
            sentiment = self.ai_module.detect_sentiment(query)
            if sentiment == 'positive':
                response_text = self._add_positive_tone(response_text)
            elif sentiment == 'negative':
                response_text = self._add_supportive_tone(response_text)
            
            self._add_to_history('bot', response_text)
            return response_text, is_fuzzy, threshold

        except Exception as e:
            print(f"Error in handle_query: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing it.", False, 0.0

    def _process_query(self, query: str) -> Tuple[str, bool, float]:
        """Process query with optimized lookup strategy using knowledge base."""
        try:
            # First attempt - direct lookup via knowledge base (includes exact matching)
            answer = self.knowledge_base.get_exact_match_answer(query)
            if self._is_valid_answer(answer):
                return answer, False, 0.0
                
            # Second attempt - expanded queries
            answer = self._try_expanded_queries(query)
            if answer is not None and answer != '' and self._is_valid_answer(answer):
                return answer, False, 0.0
                
            # Third attempt - fuzzy matching via knowledge base
            fuzzy_result = self.knowledge_base.fuzzy_match_with_thresholds(query)
            if fuzzy_result is not None:
                # Fixed: Ensure we have a valid tuple before unpacking
                if isinstance(fuzzy_result, tuple) and len(fuzzy_result) == 2:
                    answer, similarity = fuzzy_result
                    return answer, True, similarity
                else:
                    print(f"Warning: fuzzy_match_with_thresholds returned invalid format: {fuzzy_result}")
                
            # Return empty string to trigger fallback
            return "", False, 0.0
            
        except Exception as e:
            print(f"Error in _process_query: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return "", False, 0.0

    def _try_expanded_queries(self, query: str) -> str:
        """Try expanded queries efficiently."""
        try:
            expanded_queries = self.ai_module.expand_query(query)
            for expanded_query in expanded_queries:
                # Use knowledge base's normalization for consistency
                cleaned_expanded = self.knowledge_base.normalize_text(expanded_query)
                # Try exact lookup first, then fuzzy if needed
                answer = self.knowledge_base.get_exact_match_answer(expanded_query)
                if self._is_valid_answer(answer):
                    return answer
            return ""
        except Exception as e:
            print(f"Error in _try_expanded_queries: {e}")
            return ""

    @staticmethod
    def _is_valid_answer(answer: str) -> bool:
        """Check if answer is valid (not a fallback message)."""
        return not (answer.startswith("I don't have specific information") or 
                   answer.startswith("I couldn't find a good match for"))

    @lru_cache(maxsize=128)
    def generate_response(self, query: str) -> str:
        """Generate fallback response with caching for common queries."""
        try:
            if not query or not query.strip():
                return "Please ask me a question!"
                
            query_lower = query.lower()
            
            # Use the knowledge base's word cleaning method for consistency
            query_words = self.knowledge_base.clean_words(query)
            
            print(f"Debug: query='{query}', cleaned_words={query_words}")  # Debug line
            
            # Greeting (check this BEFORE time to avoid conflicts)
            if self.GREETING_WORDS & query_words:
                return "Hello! I'm an AI chatbot here to help answer your questions. What would you like to know?"
            
            # Time query
            if self.TIME_WORDS & query_words:
                return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
            
            # Date query  
            if self.DATE_WORDS & query_words:
                return f"Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}."
            
            # Goodbye
            if self.GOODBYE_WORDS & query_words:
                return "Goodbye! Feel free to come back anytime if you have more questions."
            
            # Help queries
            if any(phrase in query_lower for phrase in self.HELP_PHRASES):
                return ("I'm an AI chatbot that can answer questions based on my knowledge base. "
                       "I can handle complex queries, maintain context, and respond intelligently.")
            
            # Identity queries
            if any(phrase in query_lower for phrase in self.IDENTITY_PHRASES):
                return ("I'm an AI chatbot trained to assist with intelligent answers. I use advanced search, "
                       "fuzzy matching, and query decomposition to help you.")
            
            return self.DEFAULT_NO_MATCH_MESSAGE

        except Exception as e:
            print(f"Error in generate_response: {e}")
            return random.choice(self.default_responses)

    def _add_positive_tone(self, response: str) -> str:
        """Add positive tone to response."""
        return f"{random.choice(self.positive_starters)} {response}"

    def _add_supportive_tone(self, response: str) -> str:
        """Add supportive tone to response."""
        return f"{random.choice(self.supportive_starters)} {response}"
    
    def _add_to_history(self, msg_type: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({
            'type': msg_type,
            'content': content,
            'timestamp': datetime.datetime.now()
        })
    
    def clear_cache(self) -> None:
        """Clear the LRU cache for generate_response."""
        self.generate_response.cache_clear()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        user_messages = sum(1 for msg in self.conversation_history if msg['type'] == 'user')
        bot_messages = sum(1 for msg in self.conversation_history if msg['type'] == 'bot')
        
        return {
            'total_messages': len(self.conversation_history),
            'user_messages': user_messages,
            'bot_messages': bot_messages,
            'cache_info': self.generate_response.cache_info()
        }

    # --- New methods to leverage knowledge base features ---

    def search_knowledge_base(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge base by keyword.
        
        Args:
            keyword (str): Keyword to search for
            limit (int): Maximum number of results
            
        Returns:
            List of matching QA pairs with relevance scores
        """
        return self.knowledge_base.search_by_keyword(keyword, limit)

    def get_similar_questions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find questions similar to the given query.
        
        Args:
            query (str): Query to find similar questions for
            limit (int): Maximum number of similar questions
            
        Returns:
            List of similar questions with similarity scores
        """
        return self.knowledge_base.get_similar_questions(query, limit)

    def add_custom_knowledge(self, question: str, answer: str, context: str = "") -> bool:
        """
        Add custom knowledge to the knowledge base.
        
        Args:
            question (str): Question text
            answer (str): Answer text
            context (str): Optional context
            
        Returns:
            bool: True if successfully added
        """
        return self.knowledge_base.add_custom_qa(question, answer, context)

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge base."""
        return self.knowledge_base.get_analytics()

    def provide_feedback(self, query: str, helpful: bool, comment: str = "") -> None:
        """
        Provide feedback on a query/answer pair.
        
        Args:
            query (str): The original query
            helpful (bool): Whether the answer was helpful
            comment (str): Optional comment
        """
        self.knowledge_base.provide_feedback(query, helpful, comment)

    def optimize_system(self) -> Dict[str, Any]:
        """
        Optimize both chatbot and knowledge base performance.
        
        Returns:
            Dictionary with optimization results
        """
        # Clear chatbot cache
        self.clear_cache()
        
        # Optimize knowledge base
        kb_optimization = self.knowledge_base.optimize_knowledge_base()
        
        # Clear knowledge base cache
        self.knowledge_base.clear_cache()
        
        return {
            'chatbot_cache_cleared': True,
            'knowledge_base_optimization': kb_optimization
        }