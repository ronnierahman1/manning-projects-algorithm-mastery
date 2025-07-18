import unittest
from src.algorithms.bfs import bfs

class TestBFS(unittest.TestCase):
    """
    Unit tests for the Breadth-First Search (BFS) algorithm.
    These tests ensure that BFS correctly finds the shortest path or
    returns an empty path if no valid path exists.
    """

    def test_bfs_finds_shortest_path(self):
        """
        Test that BFS correctly finds the shortest path in a simple 3x3 grid.
        The grid has one possible clear path from start to goal.
        """
        grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        start = (0, 0)
        goal = (2, 2)
        expected_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        self.assertEqual(bfs(grid, start, goal), expected_path)

    def test_bfs_returns_empty_when_no_path(self):
        """
        Test that BFS returns an empty list when the goal is unreachable
        due to surrounding obstacles.
        """
        grid = [
            [0, 1],
            [1, 0]
        ]
        start = (0, 0)
        goal = (1, 1)
        expected_path = []
        self.assertEqual(bfs(grid, start, goal), expected_path)

    def test_bfs_same_start_and_goal(self):
        """
        Test that BFS returns a path containing only the start/goal node
        when the start and goal positions are the same.
        """
        grid = [
            [0, 0],
            [0, 0]
        ]
        start = (1, 1)
        goal = (1, 1)
        expected_path = [(1, 1)]
        self.assertEqual(bfs(grid, start, goal), expected_path)

if __name__ == '__main__':
    # Run all unit tests
    unittest.main()
