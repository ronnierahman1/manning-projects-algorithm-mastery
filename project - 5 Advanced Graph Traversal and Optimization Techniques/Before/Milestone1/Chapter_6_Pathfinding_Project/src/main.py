import pygame
import json
import os
from algorithms.bfs import bfs

def load_terrain_data(filepath):
    """
    Load terrain grid data from a JSON file.

    Args:
        filepath (str): Path to the terrain data JSON file.

    Returns:
        dict: Dictionary of grids keyed by level name.
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def draw_grids(screen, grids_info, cell_size=40, padding=40):
    """
    Draw multiple grids side-by-side with their BFS paths visualized.

    Args:
        screen (pygame.Surface): The screen to draw on.
        grids_info (list): List of (level, grid, path) tuples.
        cell_size (int): Size of each square cell in pixels.
        padding (int): Space between grids.
    """
    font = pygame.font.SysFont(None, 24)

    for idx, (level, grid, path) in enumerate(grids_info):
        rows, cols = len(grid), len(grid[0])

        # Step 1: Calculate X and Y position based on index
        x_offset = padding + (cols * cell_size + padding) * idx
        y_offset = padding

        # Step 2: Draw grid title (e.g., "Easy - BFS")
        title_surface = font.render(f"{level} - BFS", True, (0, 0, 0))
        screen.blit(title_surface, (x_offset, y_offset - 25))

        # Step 3: Draw grid cells
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
                color = (200, 200, 200) if grid[r][c] == 0 else (50, 50, 50)  # white = open, black = wall
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)  # border

        # Step 4: Draw BFS path in green
        for node in path:
            rect = pygame.Rect(x_offset + node[1] * cell_size, y_offset + node[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 200, 0), rect)


def main():
    """
    Visualize BFS pathfinding results for Easy, Medium, and Hard terrain levels
    by loading data from a JSON file and rendering it in Pygame.
    """
    pygame.init()

    # Step 1: Load terrain data from JSON
    terrain_file = os.path.join("data", "terrain_data.json")
    terrain = load_terrain_data(terrain_file)

    start, goal = (0, 0), (4, 4)

    # Step 2: Prepare list of (level, grid, bfs_path) for visualization
    grids_info = []
    for level, grid in terrain.items():
        # Call your bfs function here and store the path
        # Write your code here
        pass

    # Step 3: Setup screen size based on number of grids
    cell_size = 40
    padding = 40
    screen_width = (cell_size * 5 + padding) * len(terrain) + padding
    screen_height = cell_size * 5 + padding * 2
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Milestone 1: BFS Pathfinding on Multiple Grids")

    # Step 4: Main Pygame event/render loop
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        # Draw grids with BFS paths
        draw_grids(screen, grids_info, cell_size, padding)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
