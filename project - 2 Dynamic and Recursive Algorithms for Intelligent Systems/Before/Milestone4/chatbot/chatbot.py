# AI Chatbot Implementation - Educational "Before" Version
# This version provides the framework and detailed guidance for implementing
# the core query processing methods

import datetime  # For timestamping conversations and providing time/date responses
import random    # For selecting random responses from predefined lists
import re        # For regular expressions (pattern matching in text)
import string    # For string operations (though not directly used in this code)
from typing import List, Tuple, Optional, Dict, Any  # Type hints for better code documentation
from functools import lru_cache  # Decorator for caching function results to improve performance

# Import custom modules that handle AI processing and knowledge management
from ai.ai_module import AIModule
from chatbot.knowledge_base import KnowledgeBase

class AIChatbot:
    """
    Main chatbot class that orchestrates query processing, response generation,
    and conversation management.
    
    Architecture Overview:
    - Uses a KnowledgeBase for storing and retrieving Q&A pairs
    - Leverages an AIModule for advanced NLP tasks (sentiment analysis, query expansion)
    - Implements multiple fallback strategies for handling queries
    - Maintains conversation history and provides analytics
    
    Key Features:
    - Exact matching for precise queries
    - Fuzzy matching for similar queries
    - Query expansion for better understanding
    - Sentiment-based response tone adjustment
    - Caching for performance optimization
    """
    
    # Class constant: Default message when no specific information is found
    # This provides a consistent fallback response across the application
    DEFAULT_NO_MATCH_MESSAGE = "I don't have specific information about that topic. Could you try rephrasing your question?"
    
    # Pre-compiled regex pattern for detecting sentence-ending punctuation
    # Compiling once at class level improves performance vs. compiling repeatedly
    PUNCTUATION_PATTERN = re.compile(r'[.?!]+$')
    
    # Frozensets for word matching - these are immutable and optimized for lookup operations
    # Using frozensets instead of lists provides O(1) lookup time vs O(n) for lists
    GREETING_WORDS = frozenset(['hello', 'hi', 'hey'])
    GOODBYE_WORDS = frozenset(['bye', 'goodbye'])
    TIME_WORDS = frozenset(['time'])
    DATE_WORDS = frozenset(['date', 'today'])
    HELP_PHRASES = frozenset(['what can you do', 'help me', 'how do you work'])
    IDENTITY_PHRASES = frozenset(['who are you', 'your name'])
    
    def __init__(self, data_path: str):
        """
        Initialize the chatbot with all necessary components.
        
        Args:
            data_path (str): Path to the knowledge base data file
            
        The initialization follows a defensive programming approach with try-catch
        to handle potential errors during component setup.
        """
        try:
            # Initialize the knowledge base - this handles loading and indexing Q&A pairs
            self.knowledge_base = KnowledgeBase(data_path)
            
            # Initialize the AI module for advanced NLP tasks
            self.ai_module = AIModule()
            
            # Create a reference to QA pairs for quick access
            # This avoids repeated method calls to access the same data
            self.qa_cache = self.knowledge_base.qa_pairs
            
            # Initialize conversation history as an empty list
            # Each entry will be a dictionary with type, content, and timestamp
            self.conversation_history = []
            
            # Pre-define response lists to avoid creating them repeatedly during runtime
            # This is a performance optimization - these lists are created once and reused
            self.default_responses = [
                "I'm here to help! What would you like to know?",
                "That's an interesting question. Could you provide more details?",
                "I'd be happy to assist you with that. Can you elaborate?",
                "Let me help you with that. What specific information are you looking for?"
            ]
            
            # Positive tone starters for when sentiment analysis detects positive queries
            # These help make the chatbot sound more engaging and enthusiastic
            self.positive_starters = [
                "Great question!", "I'm happy to help!", "Excellent!", "You're on the right track!",
                "That's a thoughtful query.", "Absolutely!", "Interesting point!", "Let's dive into that.",
                "That's an insightful question!", "You've picked a great topic."
            ]
            
            # Supportive tone starters for when sentiment analysis detects negative/concerned queries
            # These help provide emotional support and reassurance
            self.supportive_starters = [
                "I understand this might be concerning.", "Let me help clarify this for you.",
                "You're doing great asking that.", "Don't worry, I've got you covered.",
                "Happy to help!", "You're asking the right person.",
                "Let's work through this together.", "That's what I'm here for.",
                "I'll do my best to guide you.", "You're not alone—I'm here to help."
            ]
            
            print("Chatbot initialized successfully!")
            
        except Exception as e:
            # If initialization fails, print the error and re-raise it
            # This follows the "fail fast" principle - better to crash early than hide errors
            print(f"Error initializing chatbot: {e}")
            raise

    def handle_query(self, query: str) -> Tuple[str, bool, float]:
        """
        Main entry point for processing user queries.
        
        This method should implement a multi-layered approach to query handling:
        1. Input validation and preprocessing
        2. Query processing through multiple strategies
        3. Sentiment analysis and tone adjustment
        4. Response formatting and history tracking
        
        Args:
            query (str): User's input query
            
        Returns:
            Tuple containing:
            - response_text (str): The generated response
            - is_fuzzy (bool): Whether fuzzy matching was used
            - threshold (float): Similarity score if fuzzy matching was used
            
        IMPLEMENTATION STEPS:
        
        Step 1: INPUT VALIDATION
        - Wrap everything in a try-catch block for error handling
        - Check if query is empty or None using early return pattern
        - If empty, return: "Please ask me a question!", False, 0.0
        - Strip whitespace from the query
        
        Step 2: CONVERSATION TRACKING
        - Add the user's query to conversation history using self._add_to_history('user', query)
        
        Step 3: CORE QUERY PROCESSING
        - Call self._process_query(query) to get response_text, is_fuzzy, threshold
        
        Step 4: FALLBACK MECHANISM
        - Check if response_text is empty or just whitespace
        - If so, call self.generate_response(query) as fallback
        - Reset is_fuzzy to False and threshold to 0.0 for fallback responses
        
        Step 5: SENTIMENT-BASED TONE ADJUSTMENT
        - Use self.ai_module.detect_sentiment(query) to analyze sentiment
        - If sentiment == 'positive': call self._add_positive_tone(response_text)
        - If sentiment == 'negative': call self._add_supportive_tone(response_text)
        
        Step 6: RESPONSE TRACKING AND RETURN
        - Add bot response to history using self._add_to_history('bot', response_text)
        - Return the tuple (response_text, is_fuzzy, threshold)
        
        Step 7: ERROR HANDLING
        - In the except block, print the error and return a graceful fallback message:
          "I apologize, but I encountered an error while processing your question. Please try rephrasing it.", False, 0.0
        """
        # TODO: Implement the method following the steps above
        pass

    def _process_query(self, query: str) -> Tuple[str, bool, float]:
        """
        Core query processing logic implementing a hierarchical matching strategy.
        
        The method should use a waterfall approach with multiple fallback levels:
        1. Exact matching (fastest, most accurate)
        2. Expanded query matching (handles variations)
        3. Fuzzy matching (handles typos and similar concepts)
        
        This approach balances accuracy with performance - exact matches are instant,
        while fuzzy matching provides flexibility at the cost of computation.
        
        Args:
            query (str): Preprocessed user query
            
        Returns:
            Tuple of (response, is_fuzzy_match, similarity_score)
            
        IMPLEMENTATION STEPS:
        
        Step 1: ERROR HANDLING SETUP
        - Wrap everything in a try-catch block
        
        Step 2: LEVEL 1 - EXACT MATCHING
        - Call self.knowledge_base.get_exact_match_answer(query) to get answer
        - Check if answer is valid using self._is_valid_answer(answer)
        - If valid, return (answer, False, 0.0)  # False = not fuzzy, 0.0 = exact match
        
        Step 3: LEVEL 2 - EXPANDED QUERY MATCHING
        - Call self._try_expanded_queries(query) to get answer
        - Check if answer is not None, not empty string, and valid using self._is_valid_answer()
        - If all conditions met, return (answer, False, 0.0)
        
        Step 4: LEVEL 3 - FUZZY MATCHING
        - Call self.knowledge_base.fuzzy_match_with_thresholds(query) to get fuzzy_result
        - Check if fuzzy_result is not None
        - Validate that fuzzy_result is a tuple with exactly 2 elements
        - If validation passes:
          * Unpack: answer, similarity = fuzzy_result
          * Return (answer, True, similarity)  # True = fuzzy match used
        - If validation fails:
          * Print warning: f"Warning: fuzzy_match_with_thresholds returned invalid format: {fuzzy_result}"
        
        Step 5: ALL STRATEGIES FAILED
        - If no strategy worked, return ("", False, 0.0) to trigger fallback
        
        Step 6: ERROR HANDLING
        - In except block:
          * Print error: f"Error in _process_query: {e}"
          * Import traceback and print full traceback for debugging
          * Return ("", False, 0.0)
        
        VALIDATION NOTE:
        The isinstance() and len() checks are crucial for defensive programming.
        They prevent crashes if the knowledge base returns unexpected data formats.
        """
        # TODO: Implement the method following the steps above
        pass

    def _try_expanded_queries(self, query: str) -> str:
        """
        Attempt to find matches using AI-generated query variations.
        
        This method leverages the AI module's query expansion capabilities
        to handle cases where users phrase questions differently than
        the knowledge base entries.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Answer if found, empty string otherwise
            
        The method tries each expanded query in sequence and returns
        the first valid answer found.
        """
        try:
            # Use AI module to generate alternative phrasings of the query
            # This might include synonyms, different word orders, or related concepts
            expanded_queries = self.ai_module.expand_query(query)
            
            # Try each expanded query until we find a match
            for expanded_query in expanded_queries:
                # Normalize the expanded query using knowledge base's normalization
                # This ensures consistency with how the knowledge base processes text
                cleaned_expanded = self.knowledge_base.normalize_text(expanded_query)
                
                # Try exact lookup first for the expanded query
                # This maintains the performance hierarchy even for expanded queries
                answer = self.knowledge_base.get_exact_match_answer(expanded_query)
                if self._is_valid_answer(answer):
                    return answer  # Return first valid answer found
                    
            # If no expanded queries yielded results, return empty string
            return ""
            
        except Exception as e:
            # Handle errors gracefully without breaking the query processing flow
            print(f"Error in _try_expanded_queries: {e}")
            return ""

    @staticmethod
    def _is_valid_answer(answer: str) -> bool:
        """
        Validate whether an answer is meaningful or just a fallback message.
        
        This static method checks if the answer is a real response rather than
        a default "not found" message. This helps the system distinguish between
        actual knowledge base answers and fallback responses.
        
        Args:
            answer (str): Answer to validate
            
        Returns:
            bool: True if answer is valid, False if it's a fallback message
            
        Static method is used because this validation logic doesn't need
        access to instance variables and can be reused independently.
        """
        # Check against known fallback message patterns
        # These patterns indicate the knowledge base couldn't find a real answer
        return not (answer.startswith("I don't have specific information") or 
                   answer.startswith("I couldn't find a good match for"))

    @lru_cache(maxsize=128)
    def generate_response(self, query: str) -> str:
        """
        Generate fallback responses for queries not found in the knowledge base.
        
        This method uses LRU (Least Recently Used) caching to improve performance
        for repeated queries. The cache stores the 128 most recent results.
        
        The method implements pattern matching for common query types:
        - Greetings and social interactions
        - Time and date requests
        - Help and identity questions
        
        Args:
            query (str): User query that couldn't be matched in knowledge base
            
        Returns:
            str: Generated response appropriate to the query type
            
        The @lru_cache decorator automatically handles caching - subsequent
        calls with the same query will return cached results instantly.
        """
        try:
            # Handle empty or whitespace-only queries
            if not query or not query.strip():
                return "Please ask me a question!"
                
            # Convert to lowercase for case-insensitive pattern matching
            query_lower = query.lower()
            
            # Use knowledge base's word cleaning for consistency
            # This ensures the same text processing pipeline is used throughout
            query_words = self.knowledge_base.clean_words(query)
            
            # Debug output to help developers understand word extraction
            print(f"Debug: query='{query}', cleaned_words={query_words}")
            
            # Pattern matching using set intersection for efficient word lookup
            # Set intersection (&) is faster than nested loops for checking word presence
            
            # Greeting detection (checked first to avoid conflicts with time queries)
            # Example: "Hello, what time is it?" should be treated as greeting primarily
            if self.GREETING_WORDS & query_words:
                return "Hello! I'm an AI chatbot here to help answer your questions. What would you like to know?"
            
            # Time query detection
            # Handles queries like "what time is it?" or "current time"
            if self.TIME_WORDS & query_words:
                return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
            
            # Date query detection
            # Handles queries like "what's today's date?" or "what date is it?"
            if self.DATE_WORDS & query_words:
                return f"Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}."
            
            # Goodbye detection
            # Provides polite closure to conversations
            if self.GOODBYE_WORDS & query_words:
                return "Goodbye! Feel free to come back anytime if you have more questions."
            
            # Help query detection using substring matching
            # This catches longer phrases that might not work with word-based matching
            if any(phrase in query_lower for phrase in self.HELP_PHRASES):
                return ("I'm an AI chatbot that can answer questions based on my knowledge base. "
                       "I can handle complex queries, maintain context, and respond intelligently.")
            
            # Identity query detection
            # Responds to questions about what the chatbot is or its capabilities
            if any(phrase in query_lower for phrase in self.IDENTITY_PHRASES):
                return ("I'm an AI chatbot trained to assist with intelligent answers. I use advanced search, "
                       "fuzzy matching, and query decomposition to help you.")
            
            # Default fallback for unrecognized patterns
            return self.DEFAULT_NO_MATCH_MESSAGE

        except Exception as e:
            # Final fallback: if pattern matching fails, use random default response
            # This ensures the chatbot never fails to respond
            print(f"Error in generate_response: {e}")
            return random.choice(self.default_responses)

    def _add_positive_tone(self, response: str) -> str:
        """
        Enhance response with positive, enthusiastic tone.
        
        This method is called when sentiment analysis detects that the user
        is in a positive mood or asking upbeat questions. It prepends
        encouraging phrases to make the interaction more engaging.
        
        Args:
            response (str): Original response text
            
        Returns:
            str: Response with positive tone prefix added
        """
        return f"{random.choice(self.positive_starters)} {response}"

    def _add_supportive_tone(self, response: str) -> str:
        """
        Enhance response with supportive, empathetic tone.
        
        This method is called when sentiment analysis detects that the user
        might be concerned, frustrated, or asking about difficult topics.
        It prepends supportive phrases to provide emotional comfort.
        
        Args:
            response (str): Original response text
            
        Returns:
            str: Response with supportive tone prefix added
        """
        return f"{random.choice(self.supportive_starters)} {response}"
    
    def _add_to_history(self, msg_type: str, content: str) -> None:
        """
        Add a message to the conversation history with metadata.
        
        This method maintains a chronological record of the conversation
        for analytics, context awareness, and debugging purposes.
        Each entry includes type, content, and timestamp.
        
        Args:
            msg_type (str): Type of message ('user' or 'bot')
            content (str): The actual message content
            
        The history structure allows for easy analysis of conversation
        patterns and user behavior.
        """
        self.conversation_history.append({
            'type': msg_type,           # Identifies speaker (user/bot)
            'content': content,         # The actual message text
            'timestamp': datetime.datetime.now()  # When the message occurred
        })
    
    def clear_cache(self) -> None:
        """
        Clear the LRU cache for the generate_response method.
        
        This method is useful for:
        - Memory management in long-running applications
        - Testing with fresh cache state
        - Forcing regeneration of responses after system updates
        
        The cache clearing is explicit rather than automatic to give
        developers control over when to optimize memory vs. performance.
        """
        self.generate_response.cache_clear()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Generate analytics about the current conversation session.
        
        This method provides insights into conversation patterns and
        system performance. It's useful for monitoring chatbot usage
        and optimizing the user experience.
        
        Returns:
            Dict containing:
            - total_messages: Overall message count
            - user_messages: Count of user inputs
            - bot_messages: Count of bot responses
            - cache_info: LRU cache performance statistics
            
        The statistics help identify conversation patterns and system efficiency.
        """
        # Count messages by type using generator expressions for efficiency
        user_messages = sum(1 for msg in self.conversation_history if msg['type'] == 'user')
        bot_messages = sum(1 for msg in self.conversation_history if msg['type'] == 'bot')
        
        return {
            'total_messages': len(self.conversation_history),
            'user_messages': user_messages,
            'bot_messages': bot_messages,
            'cache_info': self.generate_response.cache_info()  # LRU cache statistics
        }

    # ============================================================================
    # Extended API Methods: These methods leverage knowledge base features
    # for advanced functionality beyond basic Q&A
    # ============================================================================

    def search_knowledge_base(self, keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for entries containing specific keywords.
        
        This method provides a way to explore the knowledge base content
        beyond exact question matching. It's useful for discovery and
        content exploration scenarios.
        
        Args:
            keyword (str): Keyword to search for in questions and answers
            limit (int): Maximum number of results to return (default: 10)
            
        Returns:
            List of dictionaries containing matching QA pairs with relevance scores
            
        This is a pass-through method that delegates to the knowledge base's
        search functionality while maintaining a consistent API.
        """
        return self.knowledge_base.search_by_keyword(keyword, limit)

    def get_similar_questions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find questions in the knowledge base that are similar to the given query.
        
        This method helps users discover related information when their
        exact question isn't in the knowledge base. It uses semantic
        similarity rather than exact keyword matching.
        
        Args:
            query (str): Query to find similar questions for
            limit (int): Maximum number of similar questions to return (default: 5)
            
        Returns:
            List of similar questions with similarity scores
            
        This feature enhances user experience by suggesting related topics
        and helping users refine their queries.
        """
        return self.knowledge_base.get_similar_questions(query, limit)

    def add_custom_knowledge(self, question: str, answer: str, context: str = "") -> bool:
        """
        Add new knowledge to the chatbot's knowledge base at runtime.
        
        This method allows dynamic expansion of the chatbot's knowledge
        without requiring system restarts or file modifications.
        It's useful for incorporating user feedback and domain-specific information.
        
        Args:
            question (str): The question text to add
            answer (str): The corresponding answer
            context (str): Optional context information (default: empty string)
            
        Returns:
            bool: True if the knowledge was successfully added, False otherwise
            
        This capability makes the chatbot adaptable and allows for
        continuous learning and improvement.
        """
        return self.knowledge_base.add_custom_qa(question, answer, context)

    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive statistics about the knowledge base.
        
        This method provides insights into the knowledge base's size,
        content distribution, and performance characteristics.
        It's valuable for system monitoring and optimization.
        
        Returns:
            Dictionary containing detailed analytics about the knowledge base
            
        The statistics help administrators understand system capacity
        and identify opportunities for optimization.
        """
        return self.knowledge_base.get_analytics()

    def provide_feedback(self, query: str, helpful: bool, comment: str = "") -> None:
        """
        Record user feedback about query responses for system improvement.
        
        This method enables continuous learning by collecting user satisfaction
        data. The feedback can be used to improve response quality and
        identify knowledge gaps.
        
        Args:
            query (str): The original query that was asked
            helpful (bool): Whether the user found the response helpful
            comment (str): Optional detailed feedback (default: empty string)
            
        This feedback system supports iterative improvement of the chatbot's
        performance and helps prioritize knowledge base updates.
        """
        self.knowledge_base.provide_feedback(query, helpful, comment)

    def optimize_system(self) -> Dict[str, Any]:
        """
        Perform comprehensive system optimization for improved performance.
        
        This method coordinates optimization across all chatbot components:
        - Clears chatbot response cache to free memory
        - Optimizes knowledge base indices and data structures
        - Resets performance counters
        
        Returns:
            Dictionary containing optimization results and statistics
            
        Regular optimization helps maintain system performance in
        long-running applications and after significant knowledge base updates.
        """
        # Clear the chatbot's LRU cache to free memory
        self.clear_cache()
        
        # Delegate knowledge base optimization to the knowledge base component
        # This may include reindexing, cache optimization, and data structure cleanup
        kb_optimization = self.knowledge_base.optimize_knowledge_base()
        
        # Clear knowledge base internal caches
        self.knowledge_base.clear_cache()
        
        # Return comprehensive optimization report
        return {
            'chatbot_cache_cleared': True,
            'knowledge_base_optimization': kb_optimization
        }