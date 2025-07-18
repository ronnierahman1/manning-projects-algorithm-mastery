from textblob import TextBlob
from collections import defaultdict
from difflib import get_close_matches
from chatbot.knowledge_base import _STOP

class AIModule:
    """
    A module containing AI-related functionalities such as sentiment analysis 
    and query expansion to enhance chatbot responses.
    """

    def __init__(self):
        # Synonym-based expansion dictionary for improving query understanding.
        self.expansion_dict = {
            "AI": ["Artificial Intelligence", "machine learning", "deep learning"],
            "ML": ["machine learning", "AI"],
            "recursion": ["recursive function", "looping"],
            "DP": ["dynamic programming", "memoization"],
            "greedy": ["greedy algorithm", "optimal local choice"]
        }

    def detect_sentiment(self, query):
        """
        Detects the sentiment of the user query using TextBlob.
        
        Parameters:
        -----------
        query : str
            The user query.

        Returns:
        --------
        str
            The detected sentiment: 'positive', 'negative', or 'neutral'.
        """
        analysis = TextBlob(query)
        polarity = analysis.sentiment.polarity

        if polarity > 0:
            return "positive"
        elif polarity < 0:
            return "negative"
        else:
            return "neutral"

    def expand_query(self, query):
        """
        Expands the given query by adding synonyms or related terms 
        based on a predefined dictionary.

        Parameters:
        -----------
        query : str
            The user query.

        Returns:
        --------
        str
            The expanded query with additional keywords.
        """
        words = query.split()
        expanded_words = []

        for word in words:
            if word in self.expansion_dict:
                expanded_words.extend(self.expansion_dict[word])
            else:
                expanded_words.append(word)

        return " ".join(expanded_words)

    # def fuzzy_match(self, query, knowledge_base: dict):
    #     """
    #     Finds the best match for a given query from the knowledge base 
    #     using partial matching techniques.

    #     Parameters:
    #     -----------
    #     query : str
    #         The user query.
    #     knowledge_base : dict
    #         A dictionary where keys are stored queries and values are responses.

    #     Returns:
    #     --------
    #     str
    #         The best-matched query from the knowledge base or None if no close match is found.
    #     """
    #     # query_lower = query.lower()
    #     # matches = [key for key in knowledge_base.keys() if query_lower in key.lower()]

    #     # if matches:
    #     #     return knowledge_base[matches[0]]  # Return the first closest match found
    #     # return None
    #     query = query.lower().strip().rstrip("?")
    #     # get_close_matches returns the *question* key(s)
    #     best = get_close_matches(query, knowledge_base.keys(), n=1, cutoff=0.55)
    #     if best:
    #         return knowledge_base[best[0]]
    #     return None

    def fuzzy_match(self, query: str, knowledge_base: dict) -> str | None:
        """
        Return the best ANSWER for a query using fuzzy matching,
        or None if we are not confident enough.
        """
        q = query.lower().strip().rstrip("?")
        close = get_close_matches(q, knowledge_base.keys(), n=3, cutoff=0.45)

        # additional safety: keep only candidates sharing ≥1 non-stopword token
        best = None
        for cand in close:
            if len(set(q.split()) & set(cand.split()) - _STOP) >= 1:
                best = cand
                break

        return knowledge_base[best] if best else None


# Instantiate AI module for use in the chatbot
ai_module = AIModule()

if __name__ == "__main__":
    # Example usage of AI module functionalities
    sample_queries = [
        "Tell me about AI",
        "Explain recursion",
        "What is the greedy method?",
        "I am feeling sad today",
        "What is DP?"
    ]

    for q in sample_queries:
        print(f"Original Query: {q}")
        print(f"Expanded Query: {ai_module.expand_query(q)}")
        print(f"Sentiment: {ai_module.detect_sentiment(q)}")
        print("-" * 50)
