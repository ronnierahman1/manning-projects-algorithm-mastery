from chatbot.chatbot import AIChatbot
from ai.ai_module import detect_sentiment
def test_chatbot():
    chatbot = AIChatbot("data/dev-v2.0.json")
    queries = [
            ("In what country is Normandy located", "France"),
            ("where is Normandy located", "France"),
            ("Normandy located", "France"),
            ("Normandy located country", "France"),
            ("What religion were the Normans", "Catholic"),
            ("religion of the Normans", "Catholic"),
            ("religion Normans", "Catholic"),
            ("What are two basic primary resources used to guage complexity?", "time and storage"),
            ("What is the kilogram-force sometimes reffered to as?", "kilopond"),
            ("In which year did the newspaper define southern California?", "1900"),
            ("Who conceptualized the piston?", "Papin"),
            ("What researcher showed that air is a necessity for combustion?", "Robert Boyle"),
            ("What will concentrated oxygen greatly speed up?", "combustion"),
            ("What characteristic of oxygen causes it to form bonds with other elements?", "electronegativity"),
            ("What did Mitsubishi rename its Forte to?", "Dodge D-50"),
            ("How much capital did Danish law require to start a company?", "200,000 Danish krone"),
            ("What does a country acquire as it develops?", "more capital"),
            ("What institution does Robert Barro hail from?", "Harvard"),
            ("Who developed the lithium-ion battery?", "John B. Goodenough"),
            ("Who led the Mongolian Borjigin clan?", "Kublai Khan"),
            ("What was the second meaning of a Chinese word for 'barracks'?", "thanks"),
            ("When was the Riemann hypothesis proposed?", "1859"),
            ("Why do you hate me?", None), # checking for a negative sentiment, so we skip checking the response.
            ("What is the time?", None),  # Time is dynamic, so we skip checking the response.
            ("What is the date?", None),  # Date is dynamic, so we skip checking the response.
    ]

    for query, expected_response in queries:
        print(f"User: {query}")
        response = chatbot.generate_response(query)
        print(f"Sentiment: {detect_sentiment(query)}")
        print(f"AIChatbot: {response}")

        if expected_response:
            assert response == expected_response, f"Expected: {expected_response}, Got: {response}"
    print("All test cases passed!")

if __name__ == "__main__":
    print("Starting chatbot tests...")
    test_chatbot()
    print("Finished running chatbot tests.")