from chatbot.chatbot import AIChatbot

def test_chatbot():
    chatbot = AIChatbot()
    queries = [
            ("what is ai", "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines that are programmed to think and learn."),
            ("explain dynamic programming", "Dynamic Programming is an optimization technique used to solve problems by breaking them down into simpler subproblems and storing the results for future use."),
            ("what is machine learning", "Machine Learning is a subset of AI that involves training algorithms to learn patterns from data and make decisions or predictions."),
            ("what is deep learning", "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze large amounts of data."),
            ("what is natural language processing", "Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and humans through natural language."),
            ("how does reinforcement learning work", "Reinforcement Learning is a type of Machine Learning where agents learn to make decisions by performing actions and receiving feedback in the form of rewards or penalties."),
            ("what are neural networks", "Neural Networks are a series of algorithms that mimic the operations of a human brain to recognize patterns and solve complex problems."),
            ("what is supervised learning", "Supervised Learning is a type of Machine Learning where the model is trained on labeled data to make predictions."),
            ("what is unsupervised learning", "Unsupervised Learning is a type of Machine Learning where the model learns patterns from unlabeled data."),
            ("what is reinforcement learning", "Reinforcement Learning involves training agents to make decisions by rewarding or penalizing actions based on outcomes."),
            ("what is a decision tree", "A Decision Tree is a predictive model that uses a tree-like graph to represent decisions and their possible outcomes."),
            ("what is overfitting", "Overfitting occurs when a Machine Learning model learns the training data too well, resulting in poor performance on unseen data."),
            ("what is underfitting", "Underfitting occurs when a Machine Learning model is too simple to capture the underlying patterns in the data."),
            ("What is the time?", None),  # Time is dynamic, so we skip checking the response.
            ("What is the date?", None),  # Date is dynamic, so we skip checking the response.
    ]

    priority_queries = [          
        ("urgent: tell me about recursion", "Recursion is a method of solving problems where a function calls itself as a subroutine to solve a smaller instance of the problem."),
        ("important: how does a greedy algorithm work", "A greedy algorithm builds up a solution piece by piece, always choosing the next piece that offers the most immediate benefit."),
        ("nested explain nested queries", "Nested queries are queries that are embedded within other queries, often used in database operations to refine data retrieval."),
        ("nested Explain dynamic programming and greedy algorithms", 
         "Breaking down your nested query: Dynamic Programming is an optimization technique used to solve problems by breaking them down into simpler subproblems and storing the results for future use. | A greedy algorithm builds up a solution piece by piece, always choosing the next piece that offers the most immediate benefit."),

    ]

    for query, expected_response in queries:
        print(f"User: {query}")
        response = chatbot.handle_query(query)
        print(f"AIChatbot: {response}")
        if expected_response:
            assert response == expected_response, f"Expected: {expected_response}, Got: {response}"
    print("All test cases passed!")

    print(f"Priority queries:")
    for query, expected_response in priority_queries:
        print(f"User: {query}")
        response = chatbot.handle_query(query)
        print(f"AIChatbot: {response}")

if __name__ == "__main__":
    print("Starting chatbot tests...")
    test_chatbot()
    print("Finished running chatbot tests.")