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
    pq = [(0, start)]  # (cost, position)
    visited = {start: None}
    total_cost = {start: 0}

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid, rows, cols):
            move_cost = cost_map.get(neighbor, 1)
            new_cost = current_cost + move_cost

            if neighbor not in total_cost or new_cost < total_cost[neighbor]:
                total_cost[neighbor] = new_cost
                visited[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))

    return reconstruct_path(visited, start, goal), total_cost.get(goal, float('inf'))

def get_neighbors(node, grid, rows, cols):
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    neighbors = []
    for dr, dc in directions:
        nr, nc = node[0] + dr, node[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbors.append((nr, nc))
    return neighbors

def reconstruct_path(visited, start, goal):
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = visited.get(node)
        if node is None:
            return []
    path.append(start)
    return path[::-1]
