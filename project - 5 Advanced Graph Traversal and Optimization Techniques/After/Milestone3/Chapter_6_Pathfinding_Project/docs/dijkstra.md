# Dijkstra’s Algorithm

**Overview**

Dijkstra’s Algorithm is a popular weighted-graph traversal algorithm that efficiently finds the shortest path between nodes considering different weights or costs associated with edges.

## Key Concepts
- Finds the shortest path from a start node to all other nodes in weighted graphs.
- Utilizes a priority queue (heap) to select the node with the lowest cost at each step.

## Algorithm Steps
1. Assign infinite distance to all nodes except the start node (set to 0).
2. Use a priority queue to keep track of nodes with the smallest known distances.
3. Iteratively select the node with the smallest known distance and update distances to adjacent nodes.
4. Repeat until the shortest path to the goal node is determined.

## Use Cases
- GPS navigation and mapping.
- Network routing and optimization.
- Traffic routing and logistics planning.
