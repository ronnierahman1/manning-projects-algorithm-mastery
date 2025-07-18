from textblob import TextBlob
from nltk.corpus import wordnet


"""
ai_module.py

Provides AI-related functionalities like query preprocessing, expansion,
and sentiment analysis.
"""

def preprocess_query(query):
    """
    Preprocesses a query by removing extra spaces and converting it to lowercase.

    Parameters:
    -----------
    query : str
        The user query.

    Returns:
    --------
    str
        Preprocessed query.
    """
    return query.strip().lower()


def expand_query(query):
    """
    Expands a query by adding synonyms or related keywords.

    Parameters:
    -----------
    query : str
        The user query.

    Returns:
    --------
    str
        Expanded query with synonyms appended.
    """
    # Tokenize the query into individual words
    words = query.split()
    expanded_words = set(words)  # Use a set to avoid duplicate words

    for word in words:
        # Get synonyms for the word from WordNet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                expanded_words.add(lemma.name().replace("_", " "))  # Add synonyms, replacing underscores with spaces

    # Join the expanded words back into a single query string
    expanded_query = " ".join(expanded_words)
    return expanded_query

def detect_sentiment(query):
    """
    Detects the sentiment of the query (e.g., positive, negative, neutral).

    Parameters:
    -----------
    query : str
        The user query.

    Returns:
    --------
    str
        Detected sentiment: 'positive', 'negative', or 'neutral'.
    """
    # Analyze the sentiment of the query
    analysis = TextBlob(query)
    polarity = analysis.sentiment.polarity

    # Classify the sentiment based on polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"