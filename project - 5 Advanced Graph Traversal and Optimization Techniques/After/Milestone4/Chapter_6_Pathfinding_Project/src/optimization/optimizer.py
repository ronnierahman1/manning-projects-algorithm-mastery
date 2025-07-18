import numpy as np

def convert_grid_to_numpy(grid):
    """
    Converts a regular 2D Python list to a NumPy array.

    Args:
        grid (list of list of int): The original grid.

    Returns:
        np.ndarray: NumPy representation of the grid.
    """
    return np.array(grid)

def generate_cost_map_numpy(grid):
    """
    Generates a cost map using NumPy for efficient processing.

    Args:
        grid (np.ndarray): A NumPy array representing the grid.

    Returns:
        dict: A dictionary mapping (row, col) -> cost.
    """
    rows, cols = grid.shape
    cost_map = {}
    for r in range(rows):
        for c in range(cols):
            # Simple formula to create variable terrain cost: 1 + row % 3 + col % 2
            cost_map[(r, c)] = 1 + (r % 3) + (c % 2)
    return cost_map
