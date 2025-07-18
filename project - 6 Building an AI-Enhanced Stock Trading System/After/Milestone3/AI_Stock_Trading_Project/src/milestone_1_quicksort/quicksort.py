# Milestone 1: Quicksort implementation
# --------------------------------------
# This module defines a quicksort function that sorts a list of items.
# The function accepts an optional "key" argument (a function) to extract the sorting value.
# It uses the divide-and-conquer approach:
#   - If the list is empty or has one element, it is already sorted.
#   - Otherwise, a pivot element is chosen (the middle element).
#   - The list is partitioned into three parts: items less than the pivot, equal to the pivot, and greater than the pivot.
#   - The function recursively sorts the 'less than' and 'greater than' partitions, then concatenates them with the 'equal to' partition.
#
# Example usage:
#   sorted_numbers = quicksort([3, 6, 8, 10, 1, 2, 1])
#   print(sorted_numbers)  # Output: [1, 1, 2, 3, 6, 8, 10]

def quicksort(arr, key=lambda x: x):
    # Base case: if the array has 0 or 1 element, return it as is
    if len(arr) <= 1:
        return arr
    # Choose pivot as the middle element
    pivot = arr[len(arr) // 2]
    # Partition the list into three parts
    left = [x for x in arr if key(x) < key(pivot)]
    middle = [x for x in arr if key(x) == key(pivot)]
    right = [x for x in arr if key(x) > key(pivot)]
    # Recursively sort left and right partitions and combine results
    return quicksort(left, key) + middle + quicksort(right, key)
