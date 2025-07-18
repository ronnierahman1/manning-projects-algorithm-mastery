import json
from difflib import get_close_matches

_STOP = {"the", "is", "a", "an", "in", "of", "what", "which", "where", "when",
        "who", "whom", "why", "how"}                     # very short stop-list


def _token_overlap(q: str, cand: str) -> float:
    """Jaccard-like score based on meaningful tokens."""
    tq = {t for t in q.split() if t not in _STOP}
    tc = {t for t in cand.split() if t not in _STOP}
    if not tq:
        return 0.0
    return len(tq & tc) / len(tq)


class KnowledgeBase:
    """
    Manages the chatbot's knowledge base by loading data from a structured file,
    retrieving answers, and handling fuzzy matching for improved query recognition.

    Features:
    - Loads a knowledge base from a JSON file.
    - Supports exact and fuzzy matching for queries.
    - Normalizes queries to improve search accuracy.
    """

    def __init__(self, data_path):
        """
        Initializes the knowledge base by loading the dataset from a JSON file.

        Parameters:
        -----------
        data_path : str
            Path to the dataset containing questions and answers.
        """
        self.data = {}
        self.load_data(data_path)

    def load_data(self, data_path):
        """
        Loads knowledge base data from a JSON file.

        Parameters:
        -----------
        data_path : str
            Path to the dataset file.
        """
        try:
            with open(data_path, "r", encoding="utf-8") as file:
                dataset = json.load(file)
                for entry in dataset["data"]:
                    for paragraph in entry.get("paragraphs", []):
                        for qa in paragraph.get("qas", []):
                            question = qa["question"].lower().strip().rstrip("?")  # Normalize questions
                            answers = [ans["text"] for ans in qa.get("answers", []) if "text" in ans]
                            if answers:
                                self.data[question] = answers[0]  # Store the first available answer
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
    

    def get_answer(self, query: str) -> str:
        """Return the best answer or a polite fallback."""
        q = query.lower().strip().rstrip("?")

        # ── 1. exact hit ─────────────────────────────────────────────────────────
        if q in self.data:
            return self.data[q]

        # ── 2. fuzzy + token-overlap hybrid  ─────────────────────────────────────
        # first pass: classic Levenshtein similarity (lower cutoff than before)
        close = get_close_matches(q, self.data.keys(), n=5, cutoff=0.45)

        # refine by token overlap, keep the candidate with the highest score
        best_ans, best_score = "", 0.0
        for cand in close:
            score =  _token_overlap(q, cand)
            if score > best_score:
                best_score, best_ans = score, self.data[cand]

        if best_score >= 0.50:          # needs at least half the meaningful words
            return best_ans

        # ── 3. fallback ─────────────────────────────────────────────────────────
        return "I'm not sure about that. Can you provide more details?"
    

    # def get_answer(self, query: str) -> str:
    #     """
    #     Retrieves an exact or close-matched answer for a given query.

    #     Parameters:
    #     -----------
    #     query : str
    #         User query to search in the knowledge base.

    #     Returns:
    #     --------
    #     str
    #         Answer if found, otherwise an indication that no match was found.
    #     """
    #     # query = query.lower().strip().rstrip("?")  # Normalize query
    #     # if query in self.data:
    #     #     return self.data[query]

    #     # # If an exact match is not found, use fuzzy matching
    #     # close_match = get_close_matches(query, self.data.keys(), n=1, cutoff=0.7)
    #     # if close_match:
    #     #     return f"I couldn't find an exact match. Did you mean: {close_match[0]}?"

    #     # return "I'm not sure about that. Can you provide more details?"

    #     query = query.lower().strip().rstrip("?")

    #     # 1) Exact hit ----------------------------------------------------------
    #     if query in self.data:
    #         return self.data[query]

    #     # 2) BEST fuzzy hit -----------------------------------------------------
    #     #    – lower cutoff (0.55) so “Normandy located country” ≈
    #     #      “in what country is normandy located”
    #     close = get_close_matches(query, self.data.keys(), n=1, cutoff=0.55)
    #     if close:
    #         return self.data[close[0]]     # ← **return the ANSWER**, not a hint

    #     # 3) Fallback -----------------------------------------------------------
    #     return "I'm not sure about that. Can you provide more details?"


# Testing the KnowledgeBase functionality
if __name__ == "__main__":
    kb = KnowledgeBase("data/dev-v2.0.json")

    test_queries = [
        "What is the capital of France?",
        "Who wrote the Harry Potter series?",
        "What is deep learning?",
        "Tell me about recursion.",
        "Where is Normandy located?",
        "When was the Declaration of Independence signed?"
    ]

    for query in test_queries:
        print(f"User: {query}")
        print(f"AIChatbot: {kb.get_answer(query)}")
        print("-" * 50)
