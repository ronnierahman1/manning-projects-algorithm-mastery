import random
from datetime import datetime
from difflib import get_close_matches


class AIChatbot:
    """
    A realistic AI chatbot that mimics dynamic and interactive conversations
    with a broad question coverage, contextual responses, and enhanced features.
    """

    def __init__(self):
        # Expanded dataset of predefined questions and responses
        self.knowledge_base = {
            "what is ai": "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines that are programmed to think and learn.",
            "explain dynamic programming": "Dynamic Programming is an optimization technique used to solve problems by breaking them down into simpler subproblems and storing the results for future use.",
            "tell me about recursion": "Recursion is a method of solving problems where a function calls itself as a subroutine to solve a smaller instance of the problem.",
            "how does a greedy algorithm work": "A greedy algorithm builds up a solution piece by piece, always choosing the next piece that offers the most immediate benefit.",
            "explain nested queries": "Nested queries are queries that are embedded within other queries, often used in database operations to refine data retrieval.",
            "what is machine learning": "Machine Learning is a subset of AI that involves training algorithms to learn patterns from data and make decisions or predictions.",
            "what is deep learning": "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze large amounts of data.",
            "what is natural language processing": "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language.",
            "how does reinforcement learning work": "Reinforcement Learning is a type of Machine Learning where agents learn to make decisions by performing actions and receiving feedback in the form of rewards or penalties.",
            "what are neural networks": "Neural Networks are a series of algorithms that mimic the operations of a human brain to recognize patterns and solve complex problems.",
            "what is supervised learning": "Supervised Learning is a type of Machine Learning where the model is trained on labeled data to make predictions.",
            "what is unsupervised learning": "Unsupervised Learning is a type of Machine Learning where the model learns patterns from unlabeled data.",
            "what is reinforcement learning": "Reinforcement Learning involves training agents to make decisions by rewarding or penalizing actions based on outcomes.",
            "what is a decision tree": "A Decision Tree is a predictive model that uses a tree-like graph to represent decisions and their possible outcomes.",
            "what is overfitting": "Overfitting occurs when a Machine Learning model learns the training data too well, resulting in poor performance on unseen data.",
            "what is underfitting": "Underfitting occurs when a Machine Learning model is too simple to capture the underlying patterns in the data."
        }

        # Normalize the knowledge base keys for consistency
        self.knowledge_base = {key.lower().strip(): value for key, value in self.knowledge_base.items()}

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
        normalized_query = query.lower().strip()
        self.conversation_history.append(("User", query))

        # Handle nested or contextual queries
        if normalized_query.startswith("nested"):
            return self.handle_nested_query(query)

        # Fuzzy match queries against the knowledge base
        close_match = get_close_matches(normalized_query, self.knowledge_base.keys(), n=1, cutoff=0.8)

        if close_match:
            response = self.knowledge_base[close_match[0]]
        else:
            # Generate a response dynamically or return a default response
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
        responses = []

        for subquery in subqueries:
            # Process each subquery using handle_query
            response = self.handle_query(subquery.strip())
            responses.append(response)

        # Combine responses into a single coherent response
        combined_response = " | ".join(responses)
        return f"Breaking down your nested query: {combined_response}"
    def generate_response(self, query):
        """
        Generates a dynamic response for queries not in the knowledge base.
        """
        if "time" in query.lower():
            # Provide the current time if the query mentions time
            return f"The current time is {datetime.now().strftime('%H:%M:%S')} on {datetime.now().strftime('%Y-%m-%d')}."
        elif "date" in query.lower():
            # Provide the current date if the query mentions date
            return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."
        else:
            # Return a random default response for unknown queries
            return random.choice(self.default_responses)

    def interact(self, queries):
        """
        Simulates a conversation by processing a list of queries.
        """
        print("AIChatbot: Hello! How can I assist you today?")
        for query in queries:
            print(f"User: {query}")
            response = self.handle_query(query)
            print(f"AIChatbot: {response}")

# Test the enhanced chatbot
def main():
    chatbot = AIChatbot()
    queries = [
        "What is AI?",
        "Explain dynamic programming.",
        "Tell me about recursion.",
        "How does a greedy algorithm work?",
        "What is unsupervised learning?",
        "What is overfitting?",
        "Tell me the time.",
        "Tell me the date.",
        "nested Explain nested queries and dynamic programming"
    ]

    chatbot.interact(queries)

if __name__ == "__main__":
    main()
