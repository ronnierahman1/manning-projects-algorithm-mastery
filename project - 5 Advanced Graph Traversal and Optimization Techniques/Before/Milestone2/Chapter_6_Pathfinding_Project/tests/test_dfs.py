import unittest
from src.algorithms.dfs import dfs

class TestDFS(unittest.TestCase):
    """
    Unit tests for the Depth-First Search (DFS) algorithm.
    """

    def test_dfs_finds_path(self):
        """
        Ensure DFS finds a path from start to goal in a basic open grid.
        """
        grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        start = (0, 0)
        goal = (2, 2)

        # Write your code here: call dfs() and check that it starts and ends correctly
        pass

    def test_dfs_returns_empty_when_no_path(self):
        """
        DFS should return empty list when the path is blocked by obstacles.
        """
        grid = [
            [0, 1],
            [1, 0]
        ]
        start = (0, 0)
        goal = (1, 1)

        # Write your code here: call dfs() and assert result is []
        pass

    def test_dfs_same_start_and_goal(self):
        """
        DFS should return a single node if start and goal are the same.
        """
        grid = [
            [0, 0],
            [0, 0]
        ]
        start = (1, 1)
        goal = (1, 1)

        # Write your code here: call dfs() and check result is [(1, 1)]
        pass

if __name__ == '__main__':
    unittest.main()
