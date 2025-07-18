
"""
binary_search.py

In this module, we will implement the Binary Search algorithm.

Theory and Explanation:
-----------------------
1. Binary Search is an efficient searching algorithm for sorted arrays/lists.
2. It compares the target value to the middle element:
   - If not equal, it decides which half to search next (because data is sorted).
3. Time Complexity: O(log n).

Why Binary Search?
------------------
- Much faster than Linear Search for large sorted datasets.
- Requires the data to be sorted beforehand by the search attribute.

Milestone 2 (Guidance):
-----------------------
Enhance your search system by integrating Binary Search. 
Remember the data must be sorted first by the same attribute.

Implementation Hints:
---------------------
- Accept a sorted list of items by a given attribute.
- Use two pointers (left = 0, right = len(list) - 1).
- While left <= right:
  * mid = (left + right) // 2
  * Compare the 'mid' element with 'value'.
  * If smaller, move left pointer; if larger, move right pointer.
- Handle duplicates if you want all matches.

Below is a skeleton code with detailed comments.
"""

def binary_search(file_records, attribute, value):
    """
    Performs Binary Search on a sorted list of file_records.

    Parameters:
    -----------
    file_records : list
        A list of FileRecord objects sorted by the specified 'attribute'.
    attribute : str
        The attribute used to sort the list (e.g., 'name', 'size').
    value : Any
        The target value to search for.

    Returns:
    --------
    list
        A list of matching FileRecord objects (handles duplicates if needed).

    Theory:
    -------
    - Binary Search is an efficient algorithm for searching sorted data.
    - It works by dividing the dataset in half and narrowing the search range iteratively.
    - Time Complexity: O(log n).

    Step-by-Step Guide:
    -------------------
    1. **Sort the Input List**:
       - Ensure the input list (`file_records`) is sorted by the specified `attribute`.
       - Sorting is a prerequisite for Binary Search.
    
    2. **Initialize Two Pointers**:
       - Use two pointers: `left` (set to 0) and `right` (set to `len(file_records) - 1`).
       - These pointers track the current search range.

    3. **Loop Until Search Range is Exhausted**:
       - While `left <= right`:
         * Calculate the middle index: `mid = (left + right) // 2`.
         * Retrieve the middle value using `getattr(file_records[mid], attribute)`.

    4. **Compare Middle Value**:
       - If `mid_val == value`:
           * Add the record at `mid` to the results.
           * Check neighboring elements (`i = mid - 1`, `j = mid + 1`) for duplicates.
       - If `mid_val < value`:
           * Narrow the search range to the right half (`left = mid + 1`).
       - If `mid_val > value`:
           * Narrow the search range to the left half (`right = mid - 1`).

    5. **Collect Matching Records**:
       - Use a loop to collect all matching records, including duplicates, around the middle index.

    6. **Return the Results**:
       - At the end of the loop, return the list of matching records.

    Hints:
    ------
    - Ensure the list is **sorted** before performing Binary Search.
    - Use `getattr` to dynamically retrieve the attribute value for comparison.
    - Think about edge cases:
        * What happens if no matches are found? Return an empty list.
        * How will you handle duplicates? Use loops to check neighboring elements.

    Example Usage:
    --------------
    Assume you have a sorted list of file objects (`file_records`):
        results = binary_search(file_records, "size", 2000)
    This will return all file records where the size equals 2000.
    """
    # Replace the following comment with the implementation:
    # Step 1: Ensure the input list is sorted by the specified attribute.
    # Step 2: Initialize two pointers, left and right.
    # Step 3: Use a while loop to iterate as long as left <= right.
    # Step 4: Calculate the middle index and retrieve its value using getattr.
    # Step 5: Compare the middle value with the target value.
    # Step 6: Add matching records to results and check for duplicates.
    # Step 7: Narrow the search range based on the comparison.
    # Step 8: Return the results list at the end.
    
    results = []  # Initialize a list to store matching records.
    # Write your code here
    return results
