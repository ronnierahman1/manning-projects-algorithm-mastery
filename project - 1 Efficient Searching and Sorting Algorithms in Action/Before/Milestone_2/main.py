#!/usr/bin/env python3
"""
main.py

This script demonstrates file searching and sorting using a custom FileManager class.
It performs the following tasks:
1. Checks if the 'smart_file_system' folder exists, creates it if not.
2. Maintains a list of dummy files (with a name, size, creation date).
3. Ensures each dummy file exists inside 'smart_file_system'; if missing, creates it.
4. Uses the FileManager class to manage file records in memory.
5. Demonstrates Linear Search, Binary Search, Selection Sort, and Optimized Sort.

Run:
----
python main.py
"""

import os
from datetime import datetime
from utils import file_manager  # Importing FileManager from file_manager.py

# ---------------------------------------------------------------------------
# Step 1: Check/Create 'smart_file_system' folder
# ---------------------------------------------------------------------------
BASE_FOLDER = "smart_file_system"
if not os.path.exists(BASE_FOLDER):
    os.mkdir(BASE_FOLDER)
    print(f"Created folder: '{BASE_FOLDER}'")
else:
    print(f"Folder '{BASE_FOLDER}' already exists.")

# ---------------------------------------------------------------------------
# Step 2: Define a list of dummy files
# ---------------------------------------------------------------------------
# Each tuple represents a file: (file_name, file_size_in_bytes, datetime_object)
initial_files = [
    ("notes.txt", 500, datetime(2023, 1, 15)),
    ("report.pdf", 2000, datetime(2022, 12, 25)),
    ("data.csv", 1500, datetime(2023, 2, 10)),
    ("image.png", 2500, datetime(2021, 6, 5)),
    ("archive.zip", 2000, datetime(2023, 1, 15)),  # Duplicate size
]

# Additional files for demonstrating larger datasets
extra_files = [
    ("video.mp4", 7000, datetime(2020, 9, 1)),
    ("presentation.pptx", 1000, datetime(2023, 6, 30)),
    ("script.py", 800, datetime(2022, 5, 15)),
]

# ---------------------------------------------------------------------------
# Step 3: Check if each file exists in 'smart_file_system'; if not, create.
# ---------------------------------------------------------------------------
def ensure_dummy_file_exists(file_name):
    """
    Ensures that the file 'file_name' exists inside 'smart_file_system'.
    If it doesn't, creates an empty dummy file.

    Parameters:
    -----------
    file_name : str
        The name of the file to check or create.

    Returns:
    --------
    None
    """
    file_path = os.path.join(BASE_FOLDER, file_name)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(f"Dummy content for {file_name}\n")
        print(f"Created dummy file: {file_path}")
    else:
        print(f"File already exists: {file_path}")

# ---------------------------------------------------------------------------
# Step 4: Main function to demonstrate searching and sorting operations
# ---------------------------------------------------------------------------
def main():
    # Step 4.1: Ensure dummy files exist in the folder
    for fname, _, _ in initial_files:
        ensure_dummy_file_exists(fname)

    # Step 4.2: Instantiate the FileManager
    manager = file_manager.FileManager()

    # Step 4.3: Add initial files to the FileManager
    for fname, fsize, fdate in initial_files:
        manager.add_file(fname, fsize, fdate)

    # -------------------------------------------------------------------------
    # Step 5: Print the unsorted list of files
    # -------------------------------------------------------------------------
    print("\n--- Unsorted Files ---")
    manager.list_files()

    # -------------------------------------------------------------------------
    # Step 6: Perform Linear Search
    # -------------------------------------------------------------------------
    print("\n--- Linear Search for size == 2000 ---")
    found_files = manager.linear_search("size", 2000)
    for file in found_files:
        print(file)

    # -------------------------------------------------------------------------
    # Step 7: Perform Binary Search
    # -------------------------------------------------------------------------
    print("\n--- Sorting by size (Selection Sort) for Binary Search demonstration ---")
    manager.selection_sort("size")
    print("\n--- Files After Sorting by Size ---")
    manager.list_files()

    print("\n--- Binary Search for size == 2000 ---")
    found_files = manager.binary_search("size", 2000)
    for file in found_files:
        print(file)

    # -------------------------------------------------------------------------
    # Step 8: Demonstrate Selection Sort
    # -------------------------------------------------------------------------
    print("\n--- Sorting by creation_date (Selection Sort) ---")
    manager.selection_sort("creation_date")
    manager.list_files()

    # -------------------------------------------------------------------------
    # Step 9: Demonstrate Optimized Sort
    # -------------------------------------------------------------------------
    print("\n--- Adding more files to demonstrate built-in sort vs selection sort ---")
    for fname, fsize, fdate in extra_files:
        ensure_dummy_file_exists(fname)
        manager.add_file(fname, fsize, fdate)

    print("\n--- Files Before Any New Sort ---")
    manager.list_files()

    print("\n--- Optimized Sort by name ---")
    manager.optimized_sort("name")
    manager.list_files()

    print("\nAll steps completed successfully!")


# ---------------------------------------------------------------------------
# Entry point for the script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
