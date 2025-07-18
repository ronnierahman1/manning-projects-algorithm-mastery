"""
optimized_sort.py

In this module, we demonstrate an optimized sorting approach, typically 
using Python's built-in sort (Timsort) or other advanced algorithms 
like Merge Sort or Quick Sort.

Theory and Explanation:
-----------------------
1. Python's built-in `sort()` uses Timsort with an average O(n log n) complexity.
2. Much faster than O(n^2) Selection Sort for larger datasets.
3. You could also implement Merge Sort or Quick Sort here to compare performance.

Milestone 4 (Guidance):
-----------------------
Explore optimization techniques for sorting large datasets. 
Compare Selection Sort with this optimized approach.

Implementation Hints:
---------------------
- Accept a list of items (file_records) and the attribute to sort by.
- Simply call file_records.sort(key=lambda x: x[attribute]) 
  or use the built-in 'sorted(file_records, key=lambda...)'.
- Compare performance with Selection Sort for large data.
"""

def optimized_sort(file_records, attribute):
    """
    Sorts a list of file_records using Python's built-in sort (Timsort).

    Parameters:
    ----------- 
    file_records : list
        A list of FileRecord objects to be sorted.
    attribute : str
        The attribute by which we want to sort the records (e.g., 'name', 'size').

    Returns:
    --------
    list
        The sorted list (in-place sort).

    Theory:
    -------
    - Python's built-in sort uses Timsort, which is a hybrid sorting algorithm 
      combining Merge Sort and Insertion Sort.
    - Timsort is highly efficient for real-world data because it takes advantage 
      of existing order in the dataset.
    - Average Time Complexity: O(n log n).

    Step-by-Step Guide:
    -------------------
    1. **Use Python's Built-In `sort()`**:
       - Call the `sort()` method on the `file_records` list.
       - Use the `key` argument to specify the attribute by which to sort.

    2. **Access Attribute Dynamically**:
       - Use `getattr(record, attribute)` to retrieve the value of the attribute dynamically.
       - This makes the function adaptable to any attribute of the `FileRecord` class.

    3. **In-Place Sorting**:
       - The `sort()` method modifies the list in place, meaning the original list is sorted directly.

    Hints:
    ------
    - Timsort is stable, meaning it preserves the relative order of records with equal attribute values.
    - This is a great choice for large datasets due to its efficiency.

    Example Usage:
    --------------
    Assume you have a list of file objects (`file_records`) with attributes like `size`:
        optimized_sort(file_records, "size")
    This will sort the file records in ascending order based on their size.
    """
    # Replace the following comment with the implementation:
    # Step 1: Call the built-in sort() method on file_records.
    # Step 2: Use the key argument to specify a function for retrieving the attribute value.
    # Step 3: Use getattr(record, attribute) to dynamically access the specified attribute.
    # Step 4: The sort() method will sort the list in place.


    # write your code here
    return file_records
