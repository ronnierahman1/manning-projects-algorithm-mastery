import json

class KnowledgeBase:
    def __init__(self, data_path):
        """
        Initializes the knowledge base from a JSON file and normalizes question keys.

        Parameters:
        ----------
        data_path : str
            Path to the JSON file containing the SQuAD-like knowledge base.
        """
        self.data = {}

        # Load the JSON data
        with open(data_path, "r", encoding="utf-8") as file:
            raw_data = json.load(file)

        # Extract and normalize questions and answers
        for entry in raw_data["data"]:
            for paragraph in entry["paragraphs"]:
                for qa in paragraph.get("qas", []):
                    # Normalize the question
                    question = qa["question"].lower().strip().rstrip("?")
                    if qa["answers"]:
                        # Use the first answer for simplicity
                        self.data[question] = qa["answers"][0]["text"]

    def get_answer(self, query):
        """
        Retrieves an answer from the normalized knowledge base.

        Parameters:
        ----------
        query : str
            The user's query.

        Returns:
        -------
        str or None
            The matched answer or None if no match is found.
        """
        # Normalize the query
        query = query.lower().strip().rstrip("?")
        return self.data.get(query, None)
