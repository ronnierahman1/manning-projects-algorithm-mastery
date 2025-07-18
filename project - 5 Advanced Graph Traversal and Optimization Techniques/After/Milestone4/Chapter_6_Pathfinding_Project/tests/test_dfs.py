import unittest
from src.algorithms.dfs import dfs

class TestDFS(unittest.TestCase):
    """
    Unit tests for the Depth-First Search (DFS) pathfinding algorithm.
    """

    def test_dfs_finds_path(self):
        """
        DFS should return a valid path from start to goal in a simple grid.
        The path is not guaranteed to be the shortest.
        """
        grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        start = (0, 0)
        goal = (2, 2)
        result = dfs(grid, start, goal)

        # Ensure it starts and ends at correct points
        self.assertTrue(result[0] == start and result[-1] == goal)

    def test_dfs_no_path(self):
        """
        DFS should return an empty path when no route exists.
        """
        grid = [
            [0, 1],
            [1, 0]
        ]
        start = (0, 0)
        goal = (1, 1)
        result = dfs(grid, start, goal)
        self.assertEqual(result, [])

    def test_dfs_same_start_and_goal(self):
        """
        DFS should return just the start/goal if they are the same.
        """
        grid = [
            [0, 0],
            [0, 0]
        ]
        start = (1, 1)
        goal = (1, 1)
        expected = [(1, 1)]
        self.assertEqual(dfs(grid, start, goal), expected)

if __name__ == '__main__':
    unittest.main()
