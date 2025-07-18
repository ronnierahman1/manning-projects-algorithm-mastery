def dfs(grid, start, goal):
    """
    Performs Depth-First Search to explore a path from start to goal.

    Args:
        grid (list of list of int): 2D grid representing the environment (0=open, 1=obstacle).
        start (tuple): Starting coordinates (row, col).
        goal (tuple): Goal coordinates (row, col).

    Returns:
        list: A list of coordinates representing the path from start to goal (not necessarily shortest),
              or an empty list if no path is found.
    """
    rows, cols = len(grid), len(grid[0])
    stack = [start]
    visited = {start: None}

    while stack:
        current = stack.pop()

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid, rows, cols):
            if neighbor not in visited:
                visited[neighbor] = current
                stack.append(neighbor)

    return reconstruct_path(visited, start, goal)

def get_neighbors(node, grid, rows, cols):
    """
    Returns valid neighbor cells (up, down, left, right) that are not obstacles.

    Args:
        node (tuple): Current cell (row, col).
        grid (list of list of int): Grid representation.
        rows (int): Number of rows.
        cols (int): Number of columns.

    Returns:
        list: List of accessible (row, col) neighbor coordinates.
    """
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = node[0] + dr, node[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(visited, start, goal):
    """
    Rebuilds the path from the visited map.

    Args:
        visited (dict): Map of nodes to their predecessors.
        start (tuple): Starting coordinate.
        goal (tuple): Ending coordinate.

    Returns:
        list: The path from start to goal (inclusive), or empty list if unreachable.
    """
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = visited.get(node)
        if node is None:
            return []
    path.append(start)
    return path[::-1]
