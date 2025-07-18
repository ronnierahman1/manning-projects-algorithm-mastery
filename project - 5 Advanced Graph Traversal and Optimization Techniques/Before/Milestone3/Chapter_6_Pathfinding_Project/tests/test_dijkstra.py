import unittest
from src.algorithms.dijkstra import dijkstra

class TestDijkstra(unittest.TestCase):
    """
    Unit tests for Dijkstra’s Algorithm.
    You will implement the dijkstra() function being tested here.
    """

    def test_dijkstra_finds_optimal_path(self):
        """
        Should find a path with the lowest total cost.
        """
        grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        start = (0, 0)
        goal = (2, 2)

        # Define a basic cost map (all costs = 1)
        cost_map = {
            (r, c): 1
            for r in range(3)
            for c in range(3)
        }

        # Write your code here
        # Use dijkstra() and assert expected path and cost
        pass

    def test_dijkstra_no_path(self):
        """
        Should return empty path and infinite cost if unreachable.
        """
        grid = [
            [0, 1],
            [1, 0]
        ]
        cost_map = {
            (0, 0): 1,
            (1, 1): 1
        }

        # Write your code here
        pass

    def test_dijkstra_same_start_goal(self):
        """
        If start == goal, path should be a single node with 0 cost.
        """
        grid = [[0]]
        cost_map = {(0, 0): 5}

        # Write your code here
        pass

if __name__ == '__main__':
    unittest.main()
