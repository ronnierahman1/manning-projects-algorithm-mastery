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

    Step by Step Guide:
    -------------------
    1. Initialize an empty list to store matching records, e.g., `results = []`.
       This list will hold all the file records that match the target value.

    2. Loop through each record in `file_records`:
       - For each record, you need to check if its attribute matches the value.
       - If `file_records` contains dictionaries:
           * Use `record[attribute]` to access the attribute.
       - If `file_records` contains objects:
           * Use `getattr(record, attribute)` to access the attribute dynamically.

    3. If the attribute value matches the target value:
       - Add the current record to the results list using `results.append(record)`.

    4. Continue looping until all records have been checked.

    5. Return the `results` list at the end of the function.

    Hints:
    ------
    - Think about edge cases:
        * What if no records match? Ensure the function returns an empty list.
        * What if the `attribute` doesn't exist? Handle this case gracefully.
    - Use `getattr` for objects to dynamically retrieve the attribute value.
    - This algorithm works for both sorted and unsorted data.

    Example Usage:
    --------------
    Assume you have a list of file objects, `file_records`, with attributes like `name` and `size`:
        results = linear_search(file_records, "size", 2000)
    This will return all file records where the size equals 2000.
    """
   # Write your code here
   # Step 1: Initialize an empty list for results: done for you here
    results = []  # Initialize an empty list to store matching records.

    # Step 2: Loop through each record in file_records
    # Step 3: Check if the attribute matches the value    
    # Step 4: Append matching records to results
    # Step 5: Return the results list

    
    return results