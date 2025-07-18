from collections import deque

def bfs(grid, start, goal):
    """
    Performs Breadth-First Search to find the shortest path in an unweighted grid.

    Args:
        grid (list of list of int): 2D grid representing the environment. 0 = open, 1 = obstacle.
        start (tuple): Starting coordinates as (row, col).
        goal (tuple): Goal coordinates as (row, col).

    Returns:
        list: A list of coordinates representing the shortest path from start to goal,
              or an empty list if no path is found.
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque([start])
    visited = {start: None}

    while queue:
        current = queue.popleft()

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid, rows, cols):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    return reconstruct_path(visited, start, goal)

def get_neighbors(node, grid, rows, cols):
    """
    Returns valid neighboring cells (up, down, left, right) from a given node.

    Args:
        node (tuple): Current position (row, col).
        grid (list of list of int): Grid representing the map.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.

    Returns:
        list: A list of valid neighboring (row, col) tuples.
    """
    directions = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
    neighbors = []
    for dr, dc in directions:
        nr, nc = node[0] + dr, node[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(visited, start, goal):
    """
    Reconstructs the path from start to goal using the visited dictionary.

    Args:
        visited (dict): Dictionary mapping each node to its predecessor.
        start (tuple): Start node (row, col).
        goal (tuple): Goal node (row, col).

    Returns:
        list: A list of coordinates from start to goal, or empty if no path.
    """
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = visited.get(node)
        if node is None:
            return []  # Goal not reachable
    path.append(start)
    return path[::-1]
