from chatbot.chatbot import AIChatbot
from tests.test_chatbot import test_chatbot

if __name__ == "__main__":
    # print("Starting chatbot tests...")
    # test_chatbot()
    # print("Finished running chatbot tests.")
    chatbot = AIChatbot("data/dev-v2.0.json")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting the chatbot. Goodbye!")
            break
        response = chatbot.handle_query(user_input)
        print(f"AIChatbot: {response}")