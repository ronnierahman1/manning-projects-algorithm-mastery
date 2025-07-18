"""
linear_search.py

In this module, we will implement the Linear Search algorithm.

Theory and Explanation:
-----------------------
1. Linear Search is the simplest searching algorithm.
2. It works by sequentially checking each element of a list/array until 
   a match is found or the end is reached.
3. Time Complexity: O(n).

Why Linear Search?
------------------
- Easy to implement.
- Works well for small datasets or unsorted data.
- No need for data to be sorted in any particular order.

Milestone 1 (Guidance):
-----------------------
Implement a function that searches for a file (or any item)
by its attribute. For instance, searching by 'name' or 'size' 
in your file records.

Implementation Hints:
---------------------
- Accept a list of items (e.g., 'file_records') and the 'value' you seek.
- Iterate over each item in the list:
  * Compare the relevant attribute with 'value'.
  * If it matches, return (or store) the result(s).
- If you want to return all matches, collect them in a list.

Below is a skeleton code with detailed comments.
Feel free to modify or enhance.
"""

def linear_search(file_records, attribute, value):
    """
    Performs Linear Search on a list of file_records.
    
    Parameters:
    -----------
    file_records : list
        A list of file objects or dictionaries containing file information.
    attribute : str
        The attribute to match on (e.g., 'name', 'size', etc.).
    value : Any
        The target value to be matched.

    Returns:
    --------
    list
        A list of records that match the given value for the specified attribute.

    Step by Step:
    -------------
    1. Initialize an empty list, say 'results'.
    2. Loop through each record in 'file_records'.
    3. For each record, check if record[attribute] == value (if dict),
       or getattr(record, attribute) if it's an object.
    4. If it matches, append to 'results'.
    5. Return 'results' at the end.
    """
    results = []  # Initialize an empty list to store matching records.

    for record in file_records:
        # Use getattr to get the value of the specified attribute from the record.
        if getattr(record, attribute) == value:
            results.append(record)

    return results