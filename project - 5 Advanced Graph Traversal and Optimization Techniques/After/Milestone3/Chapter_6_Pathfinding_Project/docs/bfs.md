# Breadth-First Search (BFS)

## Overview

Breadth-First Search (BFS) is a graph traversal algorithm used to find the shortest path between two points in an unweighted graph. It explores all neighbors at the current depth before moving on to nodes at the next depth level. In a grid-based environment, BFS is highly effective at finding the shortest number of moves from a start to a goal cell, avoiding obstacles.

## Key Concepts

- Uses a **queue** (FIFO) to manage the traversal.
- Guarantees the **shortest path** in unweighted graphs.
- Tracks visited nodes to prevent infinite loops or revisits.

## Implementation Details

The BFS implementation includes:
- `bfs(grid, start, goal)`: Orchestrates the BFS and returns the path.
- `get_neighbors(node, grid, rows, cols)`: Identifies valid moves from a given cell.
- `reconstruct_path(visited, start, goal)`: Rebuilds the shortest path after BFS completes.

## Use Cases

- Pathfinding in 2D grids (games, robotics, simulations).
- Social networking algorithms (shortest connection path).
- Broadcasting in networks.

## Time and Space Complexity

- **Time Complexity**: O(N), where N is the total number of cells in the grid.
- **Space Complexity**: O(N), for the queue and visited tracking.

## Example

Visual output shows BFS path in green:
- `S`: Start
- `G`: Goal
- `█`: Obstacle
- `.`: BFS Path

The path is guaranteed to be the shortest in number of steps.
