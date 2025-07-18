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

    Steps:
    ------
    1. Ensure the input list is sorted by the specified attribute.
    2. Use two pointers (left and right) to track the search range.
    3. Calculate the middle index and compare its attribute value with the target value.
    4. Narrow the search range based on the comparison.
    5. Collect all matching records, including duplicates.
    """
    results = []  # Initialize a list to store matching records.
    left = 0  # Start pointer
    right = len(file_records) - 1  # End pointer

    while left <= right:
        # Calculate the middle index
        mid = (left + right) // 2

        # Use getattr to access the value of the specified attribute
        mid_val = getattr(file_records[mid], attribute)

        if mid_val == value:
            # Found a match - add it to results
            results.append(file_records[mid])

            # Check neighbors for duplicates
            i = mid - 1
            while i >= 0 and getattr(file_records[i], attribute) == value:
                results.append(file_records[i])
                i -= 1

            j = mid + 1
            while j < len(file_records) and getattr(file_records[j], attribute) == value:
                results.append(file_records[j])
                j += 1

            break  # Exit loop after collecting all matches

        elif mid_val < value:
            # Target is in the right half
            left = mid + 1
        else:
            # Target is in the left half
            right = mid - 1

    return results
