from collections import deque

def bfs(grid, start, goal):
    """
    Perform Breadth-First Search to find the shortest path from start to goal.

    Args:
        grid (list of list of int): 2D grid representing the map. 0 = open, 1 = obstacle.
        start (tuple): Starting position (row, col).
        goal (tuple): Goal position (row, col).

    Returns:
        list: A list of (row, col) tuples forming the shortest path, or an empty list if no path found.
    """
    rows, cols = len(grid), len(grid[0])

    # Step 1: Initialize the queue with the start node
    # Step 2: Create a dictionary to track visited nodes and their predecessors
    # Step 3: Loop while the queue is not empty
    #   - Dequeue the current node
    #   - If it's the goal, break
    #   - For each neighbor of the current node:
    #       - If not visited, mark as visited and enqueue it
    # Step 4: Reconstruct and return the path from visited data

    # Write your code here
    pass


def get_neighbors(node, grid, rows, cols):
    """
    Get all valid neighboring cells (up, down, left, right) that are not blocked.

    Args:
        node (tuple): Current position (row, col).
        grid (list of list of int): Grid representing the map.
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        list: List of valid neighbor (row, col) tuples.
    """
    # Step 1: Define the directions (up, down, left, right)
    # Step 2: Loop through directions and check bounds and obstacles
    # Step 3: Return the list of valid neighbors

    # Write your code here
    pass


def reconstruct_path(visited, start, goal):
    """
    Reconstruct the path from the start node to the goal using the visited dictionary.

    Args:
        visited (dict): Mapping from node to its predecessor.
        start (tuple): Start position (row, col).
        goal (tuple): Goal position (row, col).

    Returns:
        list: The path from start to goal (inclusive), or an empty list if unreachable.
    """
    # Step 1: Initialize an empty list to build the path
    # Step 2: Start from the goal node and work backwards to the start
    # Step 3: If the goal was never reached, return empty list
    # Step 4: Reverse and return the path

    # Write your code here
    pass
