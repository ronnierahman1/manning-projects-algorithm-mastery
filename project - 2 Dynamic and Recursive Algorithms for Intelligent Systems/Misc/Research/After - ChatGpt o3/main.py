from chatbot.chatbot import AIChatbot
from milestones.recursive_handling import RecursiveHandling
from milestones.dynamic_context import DynamicContext
from milestones.greedy_priority import GreedyPriority


def main():
    """
    Main function to initialize the chatbot and provide an interactive chat experience.
    - Loads the knowledge base.
    - Handles nested queries using recursion.
    - Uses dynamic programming to store context-aware responses.
    - Implements a greedy algorithm to prioritize certain queries for optimized response time.
    """

    # Initialize chatbot and milestone components
    chatbot = AIChatbot("data/dev-v2.0.json")
    recursive_handler = RecursiveHandling(chatbot)
    dynamic_context = DynamicContext()
    greedy_priority = GreedyPriority()

    print("AIChatbot: Hello! Ask me anything, or type 'exit' to quit.")

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("AIChatbot: Goodbye!")
            break
        
        # Handle nested queries using recursion
        if user_input.lower().startswith("nested"):
            response = recursive_handler.handle_recursive_query(user_input)
        else:
            # Assign priority using Greedy Algorithm
            priority = greedy_priority.get_priority(user_input)

            # Process query with dynamic context caching
            cached_response = dynamic_context.retrieve_from_cache(user_input)
            if cached_response:
                response = cached_response
            else:
                response = chatbot.handle_query(user_input)
                dynamic_context.store_in_cache(user_input, response)

        print(f"AIChatbot: {response}")


if __name__ == "__main__":
    main()
