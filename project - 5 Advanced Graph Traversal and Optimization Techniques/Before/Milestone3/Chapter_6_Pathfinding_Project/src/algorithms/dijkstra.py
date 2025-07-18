import heapq

def dijkstra(grid, start, goal, cost_map):
    """
    Performs Dijkstra’s Algorithm to find the optimal path from start to goal in a weighted grid.

    Args:
        grid (list of list of int): 2D grid (0 = walkable, 1 = wall).
        start (tuple): Starting coordinate (row, col).
        goal (tuple): Goal coordinate (row, col).
        cost_map (dict): Dictionary with cost for each cell (row, col) as key.

    Returns:
        tuple: (path as list of coordinates, total cost as int or float)
    """
    rows, cols = len(grid), len(grid[0])

    # Step 1: Initialize a priority queue with (0, start)
    # Step 2: Initialize a visited dictionary to reconstruct the path
    # Step 3: Initialize a total_cost dictionary with start node cost 0

    # Step 4: While the priority queue is not empty:
    #   - Pop the node with the lowest cost
    #   - If it's the goal, break
    #   - For each valid neighbor:
    #       - Calculate new cost
    #       - If new cost is lower than previously recorded:
    #           - Update cost and add to the priority queue

    # Step 5: Reconstruct the path and return it along with total cost

    # Write your code here
    pass

def get_neighbors(node, grid, rows, cols):
    """
    Returns walkable neighbors of a node (up/down/left/right).

    Args:
        node (tuple): Current node (row, col)
        grid (list of list of int): 2D map
        rows (int): Number of rows
        cols (int): Number of columns

    Returns:
        list: List of (row, col) neighbor positions
    """
    # Write your code here
    pass

def reconstruct_path(visited, start, goal):
    """
    Rebuilds the path from start to goal using the visited dictionary.

    Args:
        visited (dict): node -> parent
        start (tuple): Starting node
        goal (tuple): Goal node

    Returns:
        list: List of path nodes in order, or empty if no path
    """
    # Write your code here
    pass
