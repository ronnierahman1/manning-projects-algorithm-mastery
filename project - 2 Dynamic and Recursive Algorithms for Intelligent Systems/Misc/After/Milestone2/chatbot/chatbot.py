import random
from datetime import datetime
from difflib import get_close_matches
from chatbot.knowledge_base import KnowledgeBase 
from ai.ai_module import expand_query
class AIChatbot:
    """
    A realistic AI chatbot that mimics dynamic and interactive conversations
    with a broad question coverage, contextual responses, and enhanced features.
    """

    def __init__(self, data_path):
        # Expanded dataset of predefined questions and responses
        self.knowledge_base = KnowledgeBase(data_path)

        # Response templates for handling unknown or general queries
        self.default_responses = [
            "I'm not sure about that. Can you provide more details?",
            "That's an interesting question. Let me think...",
            "I'm here to help, but I might need more information to give you a better answer."
        ]

        # Context tracking for multi-level conversations
        self.conversation_history = []

    def handle_query(self, query):
        """
        Processes the user's query and generates a response.
        """
        # normalized_query = query.lower().strip()
        # self.conversation_history.append(("User", query))

        # # Handle nested or contextual queries
        # if normalized_query.startswith("nested"):
        #     return self.handle_nested_query(query)

        # response = self.knowledge_base.get_answer(normalized_query)
        # self.conversation_history.append(("AIChatbot", response))
        # return response


        """
        Processes the user's query and generates a response with fuzzy and partial matching.
        """
        normalized_query = query.lower().strip()
        self.conversation_history.append(("User", query))

        # Handle nested or contextual queries
        if normalized_query.startswith("nested"):
            return self.handle_nested_query(query)

        # Check for an exact match in the knowledge base
        if normalized_query in self.knowledge_base.data:
            response = self.knowledge_base.get_answer(normalized_query)
            self.conversation_history.append(("AIChatbot", response))
            return response

        # Fuzzy matching to find close matches
        all_questions = list(self.knowledge_base.data.keys())
        close_matches = get_close_matches(normalized_query, all_questions, n=3, cutoff=0.5)

        if close_matches:
            # Return the closest match response
            best_match = close_matches[0]
            response = self.knowledge_base.get_answer(best_match)
            self.conversation_history.append(("AIChatbot", response))
            return response

        # Partial matching: Check if the query is a substring of any knowledge base question
        for kb_question in all_questions:
            if normalized_query in kb_question:
                response = self.knowledge_base.get_answer(kb_question)
                self.conversation_history.append(("AIChatbot", response))
                return response

        # If no match is found, return a default response
        response = self.generate_response(query)
        self.conversation_history.append(("AIChatbot", response))
        return response

    def handle_nested_query(self, query):
        """
        Handles nested queries by parsing and responding to each part step by step.
        """
        # Remove the "nested" keyword and clean up the query
        query = query.replace("nested", "").strip()

        # Split the query into potential subqueries
        subqueries = query.split(" and ")
        responses = [self.handle_query(subquery.strip()) for subquery in subqueries]
        combined_response = " | ".join(responses)
        return f"Breaking down your nested query: {combined_response}"

    def generate_response(self, query):
        """
        Generates a dynamic response for queries not in the knowledge base.
        """
        if "the time" in query.lower():
            # Provide the current time if the query mentions time
            return f"The current time is {datetime.now().strftime('%H:%M:%S')} on {datetime.now().strftime('%Y-%m-%d')}."
        elif "the date" in query.lower():
            # Provide the current date if the query mentions date
            return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        else:
            # Return a random default response for unknown queries
            return self.handle_query(query)

    def interact(self, queries):
        """
        Simulates a conversation by processing a list of queries.
        """
        print("AIChatbot: Hello! How can I assist you today?")
        for query in queries:
            print(f"User: {query}")
            response = self.handle_query(query)
            print(f"AIChatbot: {response}")

