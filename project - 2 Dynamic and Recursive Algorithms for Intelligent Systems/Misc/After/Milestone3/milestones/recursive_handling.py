"""
recursive_handling.py

Handles nested user queries recursively.
"""

from milestones.greedy_priority import get_priority

def handle_recursive_query(query, context_cache, generate_response):
    """
    Handles queries recursively, prioritizing subqueries and caching results.

    Parameters:
    -----------
    query : str
        The user query to process.
    context_cache : DynamicContextCache
        The cache object for storing and retrieving responses.
    generate_response : callable
        A function that generates a response for a query.

    Returns:
    --------
    str
        The chatbot's response to the query.
    """
    # Check the cache first
    cached_response = context_cache.get(query)
    if cached_response:
        return cached_response

    # Parse query into subqueries
    subqueries = query.split(" and ")

    # Handle each subquery based on priority
    sorted_subqueries = sorted(subqueries, key=get_priority)
    responses = [generate_response(subquery.strip()) for subquery in sorted_subqueries]

    # Combine responses and cache the result
    combined_response = " | ".join(responses)
    context_cache.set(query, combined_response)
    return combined_response
