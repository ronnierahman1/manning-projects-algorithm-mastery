import unittest
from src.algorithms.dijkstra import dijkstra

class TestDijkstra(unittest.TestCase):
    """
    Unit tests for Dijkstra’s Algorithm to verify pathfinding and cost correctness.
    """

    def test_dijkstra_path_and_cost(self):
        grid = [
            [0, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ]
        start = (0, 0)
        goal = (2, 2)
        cost_map = {
            (0,0): 1, (0,1): 1, (0,2): 1,
            (1,2): 1, (2,2): 1, (2,1): 1, (2,0): 1
        }

        path, total_cost = dijkstra(grid, start, goal, cost_map)
        print("Path:", path)
        print("Total cost:", total_cost)
        expected_path = [(0,0), (0,1), (0,2), (1,2), (2,2)]
        expected_cost = 4
        self.assertEqual(path, expected_path)
        self.assertEqual(total_cost, expected_cost)

    def test_dijkstra_no_path(self):
        grid = [
            [0, 1],
            [1, 0]
        ]
        cost_map = {}
        path, cost = dijkstra(grid, (0, 0), (1, 1), cost_map)
        self.assertEqual(path, [])
        self.assertEqual(cost, float('inf'))

    def test_dijkstra_same_start_goal(self):
        grid = [
            [0]
        ]
        cost_map = {(0, 0): 3}
        path, cost = dijkstra(grid, (0, 0), (0, 0), cost_map)
        self.assertEqual(path, [(0, 0)])
        self.assertEqual(cost, 0)

if __name__ == '__main__':
    unittest.main()
