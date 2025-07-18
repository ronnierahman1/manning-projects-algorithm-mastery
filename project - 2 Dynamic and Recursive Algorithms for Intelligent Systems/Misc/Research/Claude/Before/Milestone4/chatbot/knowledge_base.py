"""
Knowledge Base Module

This module is responsible for managing the AI chatbot's knowledge base.
It includes functionality for:
- Loading and parsing a dataset in SQuAD-style JSON format
- Creating fallback sample data if the file is missing or malformed
- Answering user queries using both exact match and fuzzy (approximate) matching

The data is expected to contain a list of questions, corresponding answers,
and optional context that explains the answer.

Milestone Alignment:
  Used in milestone 4 as the core data backend.
"""

import json
import difflib
import os


class KnowledgeBase:
    """
    Manages the knowledge base for the chatbot.
    Loads QA data and provides answers to user queries via exact or approximate matching.
    """

    def __init__(self, data_path: str):
        """
        Initialize the knowledge base by loading data from a given file path.
        
        Args:
            data_path (str): Full path to the SQuAD-style JSON file.
        """
        self.qa_pairs = []             # Internal list to store (question, answer, context)
        self.data_path = data_path     # Save the file path
        self.load_data(data_path)      # Attempt to load the data on initialization

    def load_data(self, data_path: str) -> None:
        """
        Loads QA data from the provided JSON file. If the file is missing or invalid,
        fallback sample QA pairs are created instead.

        Args:
            data_path (str): Path to the JSON file containing QA data.
        """
        try:
            # === Check if file exists ===
            if not os.path.exists(data_path):
                print(f"⚠️ Warning: Data file not found at {data_path}")
                self._create_sample_data()
                return

            # === Load JSON data ===
            with open(data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # === Parse SQuAD-format JSON ===
            if 'data' in data:
                for article in data['data']:
                    for paragraph in article.get('paragraphs', []):
                        context = paragraph.get('context', '')
                        for qa in paragraph.get('qas', []):
                            question = qa.get('question', '').strip()
                            if not question:
                                continue

                            # Handle both answerable and unanswerable types
                            answers = qa.get('answers', [])
                            if answers and len(answers) > 0:
                                answer = answers[0].get('text', '').strip()
                            else:
                                # For unanswerable questions, optionally fall back to context
                                answer = (
                                    f"Based on the context: {context[:200]}..." 
                                    if context else 
                                    "I don't have enough information to answer that question."
                                )

                            # Store the parsed QA pair
                            if question and answer:
                                self.qa_pairs.append({
                                    'question': question,
                                    'answer': answer,
                                    'context': context
                                })

            print(f"✅ Loaded {len(self.qa_pairs)} QA pairs from knowledge base.")

        except FileNotFoundError:
            print(f"❌ Error: File not found: {data_path}")
            self._create_sample_data()

        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON: {e}")
            self._create_sample_data()

        except Exception as e:
            print(f"❌ Unexpected error loading data: {e}")
            self._create_sample_data()

    def _create_sample_data(self) -> None:
        """
        Creates a small hardcoded dataset of QA pairs if no external file is found or readable.
        Useful for development, testing, or fallback behavior.
        """
        self.qa_pairs = [
            {
                'question': 'What is artificial intelligence?',
                'answer': 'Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.',
                'context': 'AI technology overview'
            },
            {
                'question': 'How does machine learning work?',
                'answer': 'Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.',
                'context': 'Machine learning fundamentals'
            },
            {
                'question': 'What is natural language processing?',
                'answer': 'Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.',
                'context': 'NLP technology'
            },
            {
                'question': 'What is deep learning?',
                'answer': 'Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data.',
                'context': 'Deep learning overview'
            },
            {
                'question': 'How do neural networks work?',
                'answer': 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information using mathematical operations.',
                'context': 'Neural network basics'
            }
        ]
        print(f"Created {len(self.qa_pairs)} sample QA pairs.")

    def get_answer(self, query: str) -> str:
        """
        Responds to a user's query by first trying an exact match, and then fuzzy matching.
        Fallback response is used if no relevant QA pair is found.

        Args:
            query (str): The user's input question

        Returns:
            str: An answer string or a fallback message
        """
        try:
            # === Basic validation ===
            if not query.strip():
                return "Please ask a specific question."

            query_lower = query.lower().strip()

            # === First Try: Exact Match ===
            for qa_pair in self.qa_pairs:
                question = qa_pair.get('question', '').lower().strip()
                if query_lower == question:
                    return qa_pair.get('answer', 'No answer available.')

            # === Then Try: Fuzzy Matching ===
            best_match = None
            best_score = 0.0

            for qa_pair in self.qa_pairs:
                question = qa_pair.get('question', '').lower().strip()
                if not question:
                    continue

                # 1. Character-level similarity
                similarity = difflib.SequenceMatcher(None, query_lower, question).ratio()

                # 2. Token overlap similarity
                query_tokens = set(query_lower.split())
                question_tokens = set(question.split())
                if query_tokens and question_tokens:
                    overlap = len(query_tokens.intersection(question_tokens))
                    token_similarity = overlap / len(query_tokens.union(question_tokens))
                    similarity = max(similarity, token_similarity)

                # Keep track of the best match
                if similarity > best_score:
                    best_score = similarity
                    best_match = qa_pair

            # === If similarity is good enough, return best match ===
            if best_match and best_score > 0.5:
                return best_match.get('answer', 'No answer available.')

            # === Fallback if no match is strong enough ===
            return f"I don't have specific information about '{query}'. Could you rephrase your question or ask about something else?"

        except Exception as e:
            print(f"❌ Error in get_answer: {e}")
            return "I'm sorry, I encountered an error while processing your question."
