import pygame
import json
import os

# Import the BFS algorithm (already implemented in Milestone 1)
from algorithms.bfs import bfs

# Import the DFS algorithm — the learner will implement this in Milestone 2
from algorithms.dfs import dfs

def load_terrain_data(filepath):
    """
    Loads grid data from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing grid definitions.

    Returns:
        dict: Dictionary of grids with difficulty levels as keys.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def draw_grids(screen, grids_info, cell_size=40, padding=40):
    """
    Draws multiple grids (BFS and DFS) side-by-side in a single Pygame window.

    Args:
        screen (pygame.Surface): The surface to draw the grids on.
        grids_info (list): A list of tuples (level, algorithm_name, grid, path).
        cell_size (int): Size of each cell in pixels.
        padding (int): Padding between grids.
    """
    font = pygame.font.SysFont(None, 24)

    for idx, (level, algo_name, grid, path) in enumerate(grids_info):
        rows, cols = len(grid), len(grid[0])

        # Calculate X and Y offset for each grid position
        x_offset = padding + (cols * cell_size + padding) * (idx % 3)
        y_offset = padding + (rows * cell_size + 3 * padding) * (idx // 3)

        # Step 1: Draw the title above each grid
        title = f"{level} - {algo_name}"
        screen.blit(font.render(title, True, (0, 0, 0)), (x_offset, y_offset - 25))

        # Step 2: Draw the grid cells
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
                color = (200, 200, 200) if grid[r][c] == 0 else (50, 50, 50)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        # Step 3: Draw the computed path (in green)
        for node in path:
            rect = pygame.Rect(x_offset + node[1] * cell_size, y_offset + node[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 200, 0), rect)

def main():
    """
    Loads grid data and visualizes BFS and DFS pathfinding results
    for Easy, Medium, and Hard difficulty levels in one window.
    """
    pygame.init()

    # Step 1: Load the terrain data from the JSON file
    terrain_file = os.path.join("data", "terrain_data.json")
    terrain = load_terrain_data(terrain_file)

    # Define start and goal points
    start, goal = (0, 0), (4, 4)

    # This list will hold all the information needed to draw the grids
    grids_info = []

    # Step 2: Loop through each difficulty level (Easy, Medium, Hard)
    for level, grid in terrain.items():
        # Step 3: Run BFS for the current grid
        bfs_path = bfs(grid, start, goal)

        # Step 4: Run DFS for the current grid
        # You will implement the dfs() function in src/algorithms/dfs.py
        # Write your code here
        dfs_path = dfs(grid, start, goal)

        # Step 5: Store both BFS and DFS results for rendering
        grids_info.append((level, "BFS", grid, bfs_path))
        #--------------------------------------------------------------------------
        # Write your code here: add the DFS grid with the same grid
        #--------------------------------------------------------------------------
        

    # Step 6: Set up the Pygame screen size
    cell_size = 40
    padding = 40
    screen_width = (cell_size * 5 + padding) * 3 + padding     # 3 grids per row
    screen_height = (cell_size * 5 + padding * 3) * 2           # 2 rows (BFS + DFS)
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Milestone 2: BFS vs DFS Pathfinding")

    # Step 7: Pygame render loop
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Step 8: Clear the screen and draw all grids
        screen.fill((255, 255, 255))
        draw_grids(screen, grids_info, cell_size, padding)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
