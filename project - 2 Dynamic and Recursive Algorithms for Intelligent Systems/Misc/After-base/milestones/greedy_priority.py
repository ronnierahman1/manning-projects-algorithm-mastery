"""
greedy_priority.py

Implements a greedy algorithm to prioritize shorter queries.
"""

def get_priority(query):
    """
    Assigns a priority level to a query based on predefined keywords.

    Parameters:
    -----------
    query : str
        The user query to be prioritized.

    Returns:
    --------
    int
        Priority level (1 = highest priority, 3 = lowest priority).
    """
    query_lower = query.lower()

    # Assign priority based on keywords
    if "urgent" in query_lower:
        return 1
    elif "important" in query_lower:
        return 2
    return 3  # Default priority for general queries
