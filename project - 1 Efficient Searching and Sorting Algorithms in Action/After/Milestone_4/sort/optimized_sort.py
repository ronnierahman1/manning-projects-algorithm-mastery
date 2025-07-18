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
    - Python's built-in sort uses Timsort, which combines Merge Sort and Insertion Sort.
    - It has an average time complexity of O(n log n), making it much faster than Selection Sort
      for large datasets.
    """
    # Use Python's built-in sort with a key function that retrieves the specified attribute.
    file_records.sort(key=lambda record: getattr(record, attribute))
    return file_records
