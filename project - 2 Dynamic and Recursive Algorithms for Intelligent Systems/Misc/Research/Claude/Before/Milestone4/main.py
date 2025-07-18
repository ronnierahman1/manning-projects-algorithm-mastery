"""
Milestone 4: Integration and testing of all components

INSTRUCTIONS:
In this file, you will integrate the core chatbot logic with dynamic programming,
recursive handling, and greedy prioritization.

You are expected to:
1. Instantiate components (chatbot, recursive handler, dynamic context, greedy priority).
2. Implement the `process_query` method to route queries through the system.
3. Implement the `interactive_mode()` function to allow live interaction via terminal.
4. Display responses and diagnostic metadata to help with debugging.
"""

import sys
import os
import time
from typing import Dict, Any

# Add the current project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TODO: Import AIChatbot from chatbot.py
# Instruction: Import the AIChatbot class from the `chatbot.chatbot` module.
# from chatbot.chatbot import AIChatbot

# TODO: Import RecursiveHandling, DynamicContext, and GreedyPriority from their respective modules
# Instruction: These are located in the milestones folder.
# from milestones.recursive_handling import RecursiveHandling
# from milestones.dynamic_context import DynamicContext
# from milestones.greedy_priority import GreedyPriority


class RecursiveAIChatbotApp:
    """
    The main application class for the Recursive AI Chatbot.
    It coordinates all modules and handles interaction.
    """

    def __init__(self, data_path: str = None):
        """
        Initialize chatbot, all modules, and stats.
        """
        if data_path is None:
            # TODO: Set a default path to dev-v2.0.json inside the data/ folder.
            # Instruction: Use os.path.join with `__file__` to get the path.
            # data_path = os.path.join(os.path.dirname(__file__), 'data', 'dev-v2.0.json')

        print("🤖 Initializing Recursive AI Chatbot...")
        print("=" * 50)

        try:
            # TODO: Initialize the AIChatbot instance with data_path
            # self.chatbot = AIChatbot(data_path)

            # TODO: Initialize RecursiveHandling using self.chatbot
            # self.recursive_handler = RecursiveHandling(self.chatbot)

            # TODO: Initialize DynamicContext
            # self.dynamic_context = DynamicContext()

            # TODO: Initialize GreedyPriority
            # self.greedy_priority = GreedyPriority()

            # Stats dictionary to track app performance
            self.session_stats = {
                'queries_processed': 0,
                'cache_hits': 0,
                'recursive_queries': 0,
                'average_response_time': 0.0,
                'total_time': 0.0
            }

            # TODO: Print a success message with QA count from self.chatbot.knowledge_base.qa_pairs
            # print("✅ Chatbot initialized successfully!")
            # print(f"📊 Knowledge base loaded with {len(self.chatbot.knowledge_base.qa_pairs)} QA pairs")

        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            raise

    def _is_complex_query(self, query: str) -> bool:
        """Detects compound queries using keywords like 'and', '&', or ';'."""
        return any(token in query.lower() for token in [" and ", ";", "&"])

    def _update_stats(self, result: Dict[str, Any]):
        """Updates the session statistics with the result of a processed query."""
        self.session_stats['queries_processed'] += 1
        self.session_stats['total_time'] += result['processing_time']
        q = self.session_stats['queries_processed']
        self.session_stats['average_response_time'] = self.session_stats['total_time'] / max(1, q)

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes a query: prioritizes it, handles cache, recursion, or standard response.
        """
        start_time = time.time()

        result = {
            'query': query,
            'response': '',
            'processing_time': 0.0,
            'used_cache': False,
            'is_recursive': False,
            'priority': 2,
            'success': True,
            'error': None
        }

        try:
            # TODO: Assign priority using greedy_priority.get_priority(query)
            # result['priority'] = self.greedy_priority.get_priority(query)

            # TODO: Try retrieving from cache via dynamic_context
            # cached_response = self.dynamic_context.retrieve_from_cache(query)
            # if cached_response:
                # result['response'] = cached_response
                # result['used_cache'] = True
                # self.session_stats['cache_hits'] += 1
            # else:
                # TODO: Check for recursive query using _is_complex_query
                # if self._is_complex_query(query):
                    # result['is_recursive'] = True
                    # result['response'] = self.recursive_handler.handle_recursive_query(query)
                    # self.session_stats['recursive_queries'] += 1
                # else:
                    # result['response'] = self.chatbot.handle_query(query)

                # TODO: Cache the response for next use
                # self.dynamic_context.store_in_cache(query, result['response'])

        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            result['response'] = "I apologize, but I encountered an error processing your query."
            print(f"❌ Error processing query: {e}")

        result['processing_time'] = time.time() - start_time
        self._update_stats(result)
        return result

    def interactive_mode(self):
        """
        Terminal loop to interact with the chatbot until the user exits.
        """
        print("\n🎯 Starting Interactive Mode")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'stats' to see session statistics")
        print("Type 'help' for available commands")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                if not user_input:
                    continue

                # TODO: Handle basic commands like quit, stats, help, clear
                # Use if/elif blocks to handle each case.

                # TODO: Call process_query and print chatbot response
                # result = self.process_query(user_input)
                # print(f"\n🤖 Chatbot: {result['response']}")

                # Optional: show diagnostics if time is long or recursive
                # if result['processing_time'] > 1.0 or result['is_recursive']:
                    # print("\n📊 Processing info:")
                    # print(f"   * Time: {result['processing_time']:.2f}s")
                    # print(f"   * Priority: {result['priority']}")
                    # print(f"   * Recursive: {'Yes' if result['is_recursive'] else 'No'}")
                    # print(f"   * Cached: {'Yes' if result['used_cache'] else 'No'}")

            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")

    def _print_stats(self):
        """Prints real-time statistics for the chatbot session."""
        print("\n📈 Session Statistics:")
        for k, v in self.session_stats.items():
            if k == 'average_response_time' or k == 'total_time':
                print(f" - {k.replace('_', ' ').title()}: {v:.2f}s")
            else:
                print(f" - {k.replace('_', ' ').title()}: {v}")

    def _print_help(self):
        """Lists help commands available to users."""
        print("\n🆘 Commands:")
        print(" - quit / exit / bye: End the session")
        print(" - stats: Show current statistics")
        print(" - help: Show this help menu")
        print(" - clear: Clear the screen")

    def _print_session_summary(self):
        """Prints session summary on chatbot exit."""
        print("\n📋 Final Summary:")
        self._print_stats()


# TODO: Launch chatbot if this file is run directly
# Instruction: Use the `if __name__ == "__main__":` pattern to call app.interactive_mode()
# if __name__ == "__main__":
    # app = RecursiveAIChatbotApp()
    # app.interactive_mode()
