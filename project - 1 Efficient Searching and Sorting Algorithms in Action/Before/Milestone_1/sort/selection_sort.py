"""
selection_sort.py

In this module, we will implement the Selection Sort algorithm.

Theory and Explanation:
-----------------------
1. Selection Sort is a simple sorting algorithm.
2. It finds the minimum element in the unsorted part of the list 
   and swaps it with the element at the beginning of that part.
3. Time Complexity: O(n^2).

Why Selection Sort?
-------------------
- Conceptually simple and easy to follow.
- Not efficient for large datasets, but good for demonstration.

Milestone 3 (Guidance):
-----------------------
Implement a function to sort files by attributes (e.g., name, size, date)
using Selection Sort.

Implementation Hints:
---------------------
- Accept a list and the attribute to sort by.
- For i in range(len(list)):
  * Set min_index = i
  * For j in range(i+1, len(list)):
    - If file_records[j][attribute] < file_records[min_index][attribute],
      update min_index
  * Swap items at i and min_index.

Below is a skeleton code with detailed comments.
"""

def selection_sort(file_records, attribute):
    """
    Sorts a list of file_records using Selection Sort.

    Parameters:
    -----------
    file_records : list
        A list of file objects or dictionaries to be sorted.
    attribute : str
        The key/attribute to sort by (e.g., 'size', 'name').

    Returns:
    --------
    list
        The same list, but sorted in-place.

    Theory:
    -------
    - Selection Sort is a simple sorting algorithm.
    - It works by repeatedly finding the minimum element from the unsorted part 
      and moving it to the beginning.
    - Time Complexity: O(n^2), making it inefficient for large datasets.

    Step-by-Step Guide:
    -------------------
    1. **Initialize the Outer Loop**:
       - Iterate through the list using an index `i` (representing the start of the unsorted portion).

    2. **Find the Minimum Element**:
       - Set `min_index = i` (assume the current element is the smallest).
       - Use an inner loop to iterate through the unsorted portion (from `i+1` to the end of the list).
       - Compare the attribute values of the elements using `getattr(record, attribute)`:
           * If a smaller value is found, update `min_index` to its index.

    3. **Swap the Minimum Element**:
       - After the inner loop finishes, check if `min_index` is different from `i`.
       - If so, swap the element at `i` with the element at `min_index`.

    4. **Repeat Until Sorted**:
       - Continue the outer loop until all elements are sorted.

    Hints:
    ------
    - Use `getattr(record, attribute)` to dynamically access the attribute value.
    - Think about edge cases:
        * What if the list is already sorted? The algorithm should still work.
        * What if the list is empty or has one element? Ensure it handles these cases gracefully.

    Example Usage:
    --------------
    Assume you have a list of file objects (`file_records`) with attributes like `size`:
        selection_sort(file_records, "size")
    This will sort the file records in ascending order based on their size.
    """
    # Replace the following comment with the implementation:
    # Step 1: Loop through each element in the list with an outer loop.
    # Step 2: Initialize the current index as the minimum index.
    # Step 3: Loop through the unsorted portion of the list with an inner loop.
    # Step 4: Compare elements to find the smallest one.
    # Step 5: Swap the smallest element with the element at the current index.
    # Step 6: Repeat for all elements in the list.


    # write your code here
    return file_records
    return file_records
