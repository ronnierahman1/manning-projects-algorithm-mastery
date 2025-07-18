"""
file_manager.py

This module provides utility functions and classes for handling 
file metadata and operations. It reuses search and sort functionalities 
from the 'search' and 'sort' modules.

Guidance:
---------
- This module demonstrates how to manage file records in memory.
- Search and sorting are performed by calling reusable functions 
  from external modules ('search' and 'sort').

Use Cases:
----------
1. Managing a list of file records with add, view, search, and sort functionalities.
2. Providing input data for external search and sorting functions.
"""

from datetime import datetime
from search import linear_search, binary_search
from sort import selection_sort, optimized_sort


# ---------------------------------------------------------------------------
# FileRecord Class
# ---------------------------------------------------------------------------
class FileRecord:
    """
    Represents a file with metadata: name, size, and creation date.

    Attributes:
    -----------
    name : str
        The name of the file (e.g., 'document.txt').
    size : int
        The size of the file in bytes (e.g., 1024).
    creation_date : datetime
        The creation date of the file (as a datetime object).
    """

    def __init__(self, name, size, creation_date):
        self.name = name
        self.size = size
        self.creation_date = creation_date

    def __repr__(self):
        """
        Provides a readable string representation of the file record.
        """
        return f"FileRecord(name='{self.name}', size={self.size}, creation_date='{self.creation_date}')"


# ---------------------------------------------------------------------------
# FileManager Class
# ---------------------------------------------------------------------------
class FileManager:
    """
    Manages a collection of FileRecord objects with functionalities for:
    - Adding file records.
    - Viewing file records.
    - Searching (linear and binary search).
    - Sorting (selection sort and optimized sort).
    """

    def __init__(self):
        # Initialize an empty list to store FileRecord objects.
        self.files = []

    def add_file(self, name, size, creation_date):
        """
        Adds a new file to the manager as a FileRecord object.

        Parameters:
        -----------
        name : str
            The file name (e.g., 'example.txt').
        size : int
            The file size in bytes (e.g., 1024).
        creation_date : datetime
            The creation date of the file.

        Returns:
        --------
        None
        """
        # Create a new FileRecord object and add it to the list.
        file_record = FileRecord(name, size, creation_date)
        self.files.append(file_record)

    def list_files(self):
        """
        Prints all file records in a user-friendly format.

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        """
        print("\n--- File Records ---")
        for idx, file in enumerate(self.files, start=1):
            print(f"{idx}. Name: {file.name}, Size: {file.size}, Created: {file.creation_date}")

    # Milestone 1: Linear Search
    def linear_search(self, attribute, value):
        """
        Performs a Linear Search for files matching a specific attribute and value.

        Parameters:
        -----------
        attribute : str
            The attribute to search for (e.g., 'name', 'size', 'creation_date').
        value : Any
            The value to match.

        Returns:
        --------
        list
            A list of matching FileRecord objects.
        """
        # Call the linear_search function from the 'search' module.
        return linear_search.linear_search(self.files, attribute, value)

    # Milestone 2: Binary Search
    def binary_search(self, attribute, value):
        """
        Performs a Binary Search for files matching a specific attribute and value.
        Requires the list to be sorted by the same attribute.

        Parameters:
        -----------
        attribute : str
            The attribute to search for (e.g., 'name', 'size', 'creation_date').
        value : Any
            The value to match.

        Returns:
        --------
        list
            A list of matching FileRecord objects.
        """
        # Call the binary_search function from the 'search' module.
        return binary_search.binary_search(self.files, attribute, value)

    # Milestone 3: Selection Sort
    def selection_sort(self, attribute):
        """
        Sorts files by a specified attribute using Selection Sort.

        Parameters:
        -----------
        attribute : str
            The attribute to sort by (e.g., 'name', 'size', 'creation_date').

        Returns:
        --------
        None
        """
        # Call the selection_sort function from the 'sort' module.
        selection_sort.selection_sort(self.files, attribute)

    # Milestone 4: Optimized Sort
    def optimized_sort(self, attribute):
        """
        Sorts files by a specified attribute using Python's built-in Timsort.

        Parameters:
        -----------
        attribute : str
            The attribute to sort by (e.g., 'name', 'size', 'creation_date').

        Returns:
        --------
        None
        """
        # Call the optimized_sort function from the 'sort' module.
        optimized_sort.optimized_sort(self.files, attribute)
