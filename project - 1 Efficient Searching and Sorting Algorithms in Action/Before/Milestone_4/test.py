import time
from search.linear_search import linear_search
from search.binary_search import binary_search
from sort.selection_sort import selection_sort
from sort.optimized_sort import optimized_sort


class FileRecord:
    """Class representing a file with attributes for testing purposes."""

    def __init__(self, name, size, date):
        self.name = name
        self.size = size
        self.date = date

    def __repr__(self):
        return f"FileRecord(name='{self.name}', size={self.size}, date='{self.date}')"


# Sample Dataset
file_records = [
    FileRecord("file1.txt", 150, "2023-11-01"),
    FileRecord("file2.jpg", 2000, "2023-10-12"),
    FileRecord("file3.pdf", 500, "2023-10-15"),
    FileRecord("file4.mp3", 8000, "2023-09-01"),
    FileRecord("file5.docx", 300, "2023-11-02")
]

# Test Linear Search
def test_linear_search():
    print("\n--- Linear Search Test ---")
    start_time = time.time()
    result = linear_search(file_records, "name", "file3.pdf")
    end_time = time.time()
    print("Linear Search Result:", result)
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

# Test Binary Search (Milestone 2)
def test_binary_search():
    print("\n--- Binary Search Test ---")
    # Sort the dataset for Binary Search to work
    sorted_records = sorted(file_records, key=lambda x: x.name.lower())
    start_time = time.time()
    query = "file3.pdf".lower()
    result = binary_search(sorted_records, "name", query)
    end_time = time.time()
    print("Binary Search Result:", result)
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

# Test Selection Sort
def test_selection_sort():
    print("\n--- Selection Sort Test ---")
    unsorted_records = file_records.copy()
    start_time = time.time()
    sorted_records = selection_sort(unsorted_records, "size")
    end_time = time.time()
    print("Selection Sort Result:", sorted_records)
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

# Test Optimized Sort (Timsort)
def test_optimized_sort():
    print("\n--- Optimized Sort Test ---")
    unsorted_records = file_records.copy()
    start_time = time.time()
    sorted_records = optimized_sort(unsorted_records, "size")
    end_time = time.time()
    print("Optimized Sort Result:", sorted_records)
    print(f"Execution Time: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    test_linear_search()
    test_binary_search()
    test_selection_sort()
    test_optimized_sort()
