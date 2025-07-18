"""
Main Chatbot Engine (Before Version for Milestone 4)

This is the scaffold version. Learners will implement:
- Knowledge base lookup
- Fuzzy matching
- Sentiment-based tone enhancement
- Nested query handling
- Fallback logic
"""

import datetime
import random
from typing import List

from ai.ai_module import AIModule
from chatbot.knowledge_base import KnowledgeBase


class AIChatbot:
    """
    Main AI Chatbot class.

    Responsibilities:
    - Accepts and processes user input.
    - Finds relevant answers from the knowledge base.
    - Handles complex queries via decomposition.
    - Adapts responses based on user sentiment.
    - Tracks conversation context.
    """

    def __init__(self, data_path: str):
        """
        Initializes the chatbot engine.

        Args:
            data_path (str): Path to the JSON-formatted knowledge base file.
        """
        try:
            self.knowledge_base = KnowledgeBase(data_path)  # Load static Q&A data
            self.ai_module = AIModule()  # Load tools for NLP processing
            self.conversation_history = []  # To track interaction over time
            self.default_responses = [
                "I'm here to help! What would you like to know?",
                "That's an interesting question. Could you provide more details?",
                "I'd be happy to assist you with that. Can you elaborate?",
                "Let me help you with that. What specific information are you looking for?"
            ]
            print("Chatbot initialized successfully!")
        except Exception as e:
            print(f"Error initializing chatbot: {e}")
            raise

    def handle_query(self, query: str) -> str:
        """
        Processes a single user query and returns an appropriate response.

        Steps:
        1. Cleans and stores query.
        2. Detects sentiment for tone adjustment.
        3. Determines if query is compound (Milestone 1).
        4. Uses:
            - Knowledge base (exact/fuzzy)
            - Synonym-expanded variations (Milestone 2)
            - Fallback response generator
        5. Adjusts response tone based on sentiment (Milestone 2).
        6. Returns final formatted response.

        Args:
            query (str): The user’s question or input.

        Returns:
            str: The chatbot’s reply.
        """
        try:
            # Step 1: Handle empty input
            # TODO: If query is empty or only spaces, return "Please ask me a question!"

            # Step 2: Clean query and store in history
            # Done for you: Strip whitespace and store query in self.conversation_history with type 'user' and current timestamp

            query = query.strip()
            self.conversation_history.append({'type': 'user', 'content': query, 'timestamp': datetime.datetime.now()})

            # Step 3: Detect sentiment
            # Done already: Use self.ai_module.detect_sentiment(query) and store the result

            # Analyze user emotion (positive/negative/neutral)
            sentiment = self.ai_module.detect_sentiment(query)

            # Step 4: Determine if query is nested (Milestone 1)
            # TODO: Use self._is_nested_query to check
            # TODO: If True, call self.handle_nested_query(query) and assign to response

            # Handle nested query path
            if self._is_nested_query(query):
                response = "" # TODO: Call the nested query handler
            # Step 5: If not nested, try regular processing
            else:
                # TODO: Use self.ai_module.expand_query(query) to get synonym-expanded queries
                # Synonym expansion: transforms input into similar versions
                expanded_queries = [] #TODO: assign self.ai_module.expand_query(query) here

                # Step 6: Try exact match from expanded queries
                # TODO: Loop through expanded queries and use self.knowledge_base.get_answer

                response = None
                for expanded_query in expanded_queries:
                    response = "" # TODO: assign self.knowledge_base.get_answer(expanded_query) here
                    # Break the loop if a match is found (not a fallback response)
                    if not response.startswith("I don't have specific information"):
                        break

                # Step 7: Try fuzzy match if no good exact match found
                # TODO: Use self.ai_module.fuzzy_match and check if confidence > 0.6

                if not response or response.startswith("I don't have specific information"):
                    fuzzy_result = "" # TODO: assign self.ai_module.fuzzy_match(query, self.knowledge_base) here
                    if fuzzy_result and fuzzy_result[2] > 0.6:
                        response = fuzzy_result[1]
                        # TODO: Append "(Note: I found a similar question...)" if fuzzy match is used
                        response += "" # TODO: assign the string here "\n\n(Note: I found a similar question and provided the best matching answer)"

                # Step 8: Use fallback rule-based response if still no match
                # TODO: Call self.generate_response(query) if needed

                # Fallback logic: use built-in rule-based response generator
                if not response or response.startswith("I don't have specific information"):
                    response = "" # TODO: Assign self.generate_response(query) here

            # Step 9: Personalize response based on sentiment (Milestone 2)
            # TODO: If sentiment is 'positive', call self._add_positive_tone(response)
            # TODO: If sentiment is 'negative', call self._add_supportive_tone(response)

            # Personalize response with tone
            if sentiment == 'positive':
                response ="" #TODO: call self._add_positive_tone(response) here and assign it to response
            elif sentiment == 'negative':
                response = "" #TODO: call self._add_supportive_tone(response) here and assign it to response 

            # Step 10: Store bot response in history
            # Done for you: Append response to self.conversation_history with type 'bot' and timestamp

            self.conversation_history.append({'type': 'bot', 'content': response, 'timestamp': datetime.datetime.now()})
            return response

        except Exception as e:
            print(f"Error in handle_query: {e}")
            return "I apologize, but I encountered an error while processing your question. Please try rephrasing it."

    def handle_nested_query(self, query: str) -> str:
        """
        Handles queries composed of multiple parts.

        Examples:
        - "What is AI and how does ML work?"
        - "Explain both deep learning; neural networks"

        Args:
            query (str): Complex, compound query.

        Returns:
            str: Multi-part response.
        """
        try:
            # Step 1: Split the complex query into parts
            # TODO: Use self._split_nested_query(query) and store result in 'parts'

            parts =[] # TODO: assign self._split_nested_query(query) here instead of []

            # Step 2: Prepare a list to store responses for each part
            responses = []
            # Step 3: Iterate through each part and generate responses
            # TODO: Loop through parts using enumerate()
            #   - Strip whitespace from each part
            #   - If the part is not empty:
            #       a. Get direct answer using self.knowledge_base.get_answer(part)
            #       b. If answer is a fallback response, try fuzzy match:
            #           - Use self.ai_module.fuzzy_match(part, self.knowledge_base)
            #           - If a result exists and confidence > 0.5, use that answer instead
            #       c. Format as "**Part i**: part\n**Answer**: answer"
            #       d. Append to 'responses' list

            for i, part in enumerate(parts):
                part = "" # TODO: strip the string using the strip() method
                if part:
                    part_response = "" #TODO: Assign the value by calling self.knowledge_base.get_answer(part)
                    if part_response.startswith("I don't have specific information"):
                        fuzzy_result = [] #TODO: Assign the value by calling fuzzy_match(part, self.knowledge_base) of AiModule
                        
                        # If fuzzy match returns a result with confidence > 0.5, use that instead
                        if fuzzy_result and fuzzy_result[2] > 0.5:
                            part_response = "" #TODO: use the result from fuzzy_result using index 1

                    responses.append(f"**Part {i+1}**: {part}\n**Answer**: {part_response}")

            return "\n\n".join(responses) if responses else "I couldn't break down your question into parts I can answer. Could you try asking each part separately?"

        except Exception as e:
            print(f"Error in handle_nested_query: {e}")
            return "I had trouble processing your complex question. Could you try breaking it down into simpler parts?"

    def generate_response(self, query: str) -> str:
        """
        Rule-based response generation if no match is found.

        Covers:
        - Greetings, farewells
        - Time/date
        - Capability or personal questions

        Args:
            query (str): User input.

        Returns:
            str: Reasonable fallback response.
        """
        try:
            query_lower = query.lower()

            # Handle time/date
            now = datetime.datetime.now()
            if 'time' in query_lower:
                return f"The current time is {now.strftime('%I:%M %p')}."
            elif 'date' in query_lower or 'today' in query_lower:
                return f"Today's date is {now.strftime('%B %d, %Y')}."

            # Greetings
            if any(word in query_lower for word in ['hello', 'hi', 'hey']):
                return "Hello! I'm an AI chatbot here to help answer your questions. What would you like to know?"

            # Farewells
            if any(word in query_lower for word in ['bye', 'goodbye']):
                return "Goodbye! Feel free to come back anytime if you have more questions. Have a great day!"

            # Capability questions
            if any(phrase in query_lower for phrase in ['what can you do', 'help me', 'how do you work']):
                return ("I'm an AI chatbot that can answer questions based on my knowledge base. "
                        "I can handle complex queries, maintain conversation context, and provide useful information.")

            # Personal ID questions
            if any(word in query_lower for word in ['who are you', 'your name']):
                return ("I'm an AI chatbot designed to help answer questions and have conversations. "
                        "I use techniques like recursion and NLP to assist users intelligently.")

            return "I don't have specific information about that topic. Could you try rephrasing your question?"

        except Exception as e:
            print(f"Error in generate_response: {e}")
            return "I'm here to help! What would you like to know?"

    def _is_nested_query(self, query: str) -> bool:
        """
        Detects compound questions using indicator keywords or delimiters.
        """
        nested_indicators = [' and ', ' & ', '; ', 'also', 'additionally', 'furthermore',
                             'what about', 'how about', 'tell me about', 'explain both']
        return any(indicator in query.lower() for indicator in nested_indicators)

    def _split_nested_query(self, query: str) -> List[str]:
        """
        Splits a nested query into individual sub-questions.
        """
        separators = [' and ', ' & ', '; ']
        parts = [query]

        for separator in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(separator))
            parts = new_parts

        return [part.strip() for part in parts if len(part.strip()) > 3]

    def _add_positive_tone(self, response: str) -> str:
        """
        Add a cheerful tone for positive sentiment detection.
        """
        starters = [
            "Great question!", "I'm happy to help!", "Excellent!", "You're on the right track!",
            "That's a thoughtful query.", "Absolutely!", "Interesting point!", "Let's dive into that.",
            "That’s an insightful question!", "You’ve picked a great topic."
        ]
        return random.choice(starters) + " " + response

    def _add_supportive_tone(self, response: str) -> str:
        """
        Add a reassuring tone for negative sentiment detection.
        """
        starters = [
            "I understand this might be concerning.", "Let me help clarify this for you.",
            "You're doing great asking that.", "Don't worry, I’ve got you covered.",
            "Happy to help!", "You're asking the right person.",
            "Let's work through this together.", "That's what I'm here for.",
            "I'll do my best to guide you.", "You're not alone—I'm here to help."
        ]
        return random.choice(starters) + " " + response
