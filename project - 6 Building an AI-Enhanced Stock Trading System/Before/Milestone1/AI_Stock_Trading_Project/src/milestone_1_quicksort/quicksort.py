def quicksort(arr, key=lambda x: x):
    """
    Instructions for implementing the quicksort function:

    1. Base Case:
       - Check if the input list 'arr' has 0 or 1 element.
       - If yes, return the list immediately (it is already sorted).

    2. Choosing a Pivot:
       - Select a pivot element from the list.
         (For example, you might choose the middle element: arr[len(arr) // 2].)

    3. Partitioning:
       - Create three sublists:
         a) 'left': containing all items where key(item) is less than key(pivot)
         b) 'middle': containing all items where key(item) is equal to key(pivot)
         c) 'right': containing all items where key(item) is greater than key(pivot)
    
    4. Recursion:
       - Recursively call quicksort on the 'left' and 'right' sublists.

    5. Concatenation:
       - Concatenate the sorted 'left' sublist, the 'middle' sublist, and the sorted 'right' sublist.
       - Return this concatenated list as the final sorted output.

    Replace the placeholder below with your complete implementation.
    """
    # Write your code here
    pass
