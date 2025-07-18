"""
Milestone 1: Recursive Query Handling

This module implements the logic needed to identify and process nested or compound queries.
It breaks down complex questions into simpler sub-questions and processes them individually
to provide informative and structured responses to the user.

The main idea is to use recursion to handle layered user input and heuristics to identify
how to split such input correctly. This mimics how advanced chatbots decompose questions.
"""

import re
from typing import List


class RecursiveHandling:
    """
    A class to process nested (recursive) user queries by breaking them into sub-questions.
    This class uses a combination of regular expressions and natural language heuristics
    to identify whether a query contains multiple parts, and then recursively answers
    each part using the chatbot’s knowledge base.

    Attributes:
        chatbot (Any): The AIChatbot instance that is responsible for answering sub-questions.
        max_recursion_depth (int): Limit to prevent infinite recursion and overly deep logic.
        recursion_patterns (List[str]): Regex patterns to match common compound query forms.
    """

    def __init__(self, chatbot):
        """
        Constructor to initialize the RecursiveHandling class.

        Args:
            chatbot (Any): A reference to the main chatbot instance. This allows delegation
                           of simple questions once a complex query has been decomposed.
        """
        self.chatbot = chatbot
        self.max_recursion_depth = 5  # Prevents infinite recursion or overly deep nesting
        self.recursion_patterns = [
            r'tell me about (.*) and (.*)',
            r'what about (.*) and (.*)',
            r'explain (.*) and (.*)',
            r'describe (.*) and (.*)',
            r'(.*) and also (.*)',
            r'(.*) as well as (.*)'
        ]

    def handle_recursive_query(self, query: str, depth: int = 0) -> str:
        """
        Primary entry point for handling recursive queries. This method determines whether
        a query is compound. If yes, it delegates to the `handle_nested_query()` method.
        If not, it asks the chatbot to respond directly.

        Args:
            query (str): The user input string (potentially complex).
            depth (int): Tracks the recursion level to prevent overflow.

        Returns:
            str: The chatbot’s response, possibly composed of multiple answers.
        """
        try:
            # Guard clause: If the recursion exceeds the max depth, abort with message
            if depth >= self.max_recursion_depth:
                return "The query is too complex. Please try breaking it into simpler parts."

            # Step 1: Check if the query is compound/nested
            if self._is_nested_query(query):
                # Step 2: If so, process recursively
                return self.handle_nested_query(query, depth + 1)
            else:
                # Step 3: If not compound, handle it as a basic single query
                return self.chatbot.knowledge_base.get_answer(query)

        except Exception as e:
            print(f"Error in handle_recursive_query: {e}")
            return "I encountered an error processing your query. Please try rephrasing it."

    def handle_nested_query(self, query: str, depth: int = 0) -> str:
        """
        This method handles compound queries by identifying their components, then
        recursively processing each sub-query and combining the responses into a
        structured final answer.

        Args:
            query (str): A nested user query, e.g., "What is AI and how does it work?"
            depth (int): The current recursion depth

        Returns:
            str: A composed answer that responds to each sub-query in the original question.
        """
        try:
            if depth >= self.max_recursion_depth:
                return "Query too complex for recursive processing."

            # Use heuristics and regex to identify sub-questions
            sub_queries = self.split_into_subqueries(query)

            if len(sub_queries) <= 1:
                # If no meaningful split is possible, treat it as a basic query
                return self.chatbot.knowledge_base.get_answer(query)

            responses = []
            for i, sub_query in enumerate(sub_queries):
                sub_query = sub_query.strip()
                if sub_query:
                    # Recurse into each identified sub-query
                    sub_response = self.handle_recursive_query(sub_query, depth + 1)
                    # Format the output nicely
                    responses.append(f"**Question {i + 1}**: {sub_query}\n**Answer**: {sub_response}")

            # Step 4: Combine all responses
            if responses:
                intro = "I'll address each part of your question:\n\n"
                return intro + "\n\n".join(responses)
            else:
                return "I couldn't break down your question into manageable parts. Could you try asking more specifically?"

        except Exception as e:
            print(f"Error in handle_nested_query: {e}")
            return "I had trouble processing your complex question. Please try asking each part separately."

    def split_into_subqueries(self, query: str) -> List[str]:
        """
        Attempts to split a complex query into its constituent sub-questions using:
        1. Predefined regex patterns
        2. Known conjunctions and delimiters
        3. Detection of multiple question types in the sentence

        Args:
            query (str): The full query text

        Returns:
            List[str]: A list of simpler, self-contained queries.
        """
        try:
            # First, attempt to extract parts using known compound patterns
            for pattern in self.recursion_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    return [group.strip() for group in groups if group.strip()]

            # Second, try to split based on conjunctions and delimiters
            conjunctions = [' and ', ' & ', '; ', ', and ', ' or ', ' plus ']
            parts = [query]  # Start with the full query as the only part

            for conj in conjunctions:
                new_parts = []
                for part in parts:
                    if conj in part:
                        new_parts.extend(part.split(conj))
                    else:
                        new_parts.append(part)
                parts = new_parts

            # Third, use question words to detect sentence boundaries
            question_words = ['what', 'how', 'why', 'where', 'when', 'who', 'which']
            sentences = re.split(r'[.!?]+', query)

            if len(sentences) > 1:
                valid_parts = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (sentence and 
                        (any(qw in sentence.lower() for qw in question_words) or 
                         len(sentence) > 10)):
                        valid_parts.append(sentence)
                if len(valid_parts) > 1:
                    return valid_parts

            # Lastly, apply minimum length filter and return cleaned parts
            clean_parts = []
            for part in parts:
                part = part.strip()
                if part and len(part) > 5:
                    clean_parts.append(part)

            return clean_parts if len(clean_parts) > 1 else [query]

        except Exception as e:
            print(f"Error in split_into_subqueries: {e}")
            return [query]

    def _is_nested_query(self, query: str) -> bool:
        """
        A heuristic checker that determines whether a query is likely to be recursive.
        Looks for keywords and conjunctions that signal multiple thoughts or clauses.

        Args:
            query (str): The user input

        Returns:
            bool: True if the input appears to be nested, False otherwise.
        """
        nested_indicators = [
            ' and ', ' & ', '; ', ' or ', ' plus ',
            'also', 'additionally', 'furthermore', 'moreover',
            'what about', 'how about', 'tell me about',
            'explain both', 'describe both', 'compare'
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in nested_indicators)
