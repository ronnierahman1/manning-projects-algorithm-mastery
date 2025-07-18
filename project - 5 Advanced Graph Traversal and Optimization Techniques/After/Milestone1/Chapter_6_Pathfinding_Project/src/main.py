import pygame
import json
import os
from algorithms.bfs import bfs

def load_terrain_data(filepath):
    """
    Loads terrain grid data for each difficulty level from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing grids keyed by difficulty level.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def draw_grids(screen, grids_info, cell_size=40, padding=40):
    """
    Draws multiple BFS pathfinding results side by side.

    Args:
        screen (pygame.Surface): Pygame screen surface to draw on.
        grids_info (list): List of tuples (level_name, grid, path).
        cell_size (int): Size of each grid cell in pixels.
        padding (int): Padding between grids.
    """
    font = pygame.font.SysFont(None, 24)

    for idx, (level, grid, path) in enumerate(grids_info):
        rows, cols = len(grid), len(grid[0])

        # Position each grid horizontally
        x_offset = padding + (cols * cell_size + padding) * idx
        y_offset = padding

        # Draw level title
        title_surface = font.render(f"{level} - BFS", True, (0, 0, 0))
        screen.blit(title_surface, (x_offset, y_offset - 25))

        # Draw grid and obstacles
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
                color = (200, 200, 200) if grid[r][c] == 0 else (50, 50, 50)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)  # border

        # Draw BFS path (in green)
        for node in path:
            rect = pygame.Rect(x_offset + node[1] * cell_size, y_offset + node[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 200, 0), rect)

def main():
    """
    Loads terrain data and displays BFS paths for Easy, Medium, and Hard grids in one window.
    """
    pygame.init()

    # Path to the terrain JSON file
    terrain_file = os.path.join("../data", "terrain_data.json")
    terrain = load_terrain_data(terrain_file)

    start, goal = (0, 0), (4, 4)

    # Collect all BFS paths per level
    grids_info = []
    for level, grid in terrain.items():
        path = bfs(grid, start, goal)
        grids_info.append((level, grid, path))

    cell_size = 40
    padding = 40
    screen_width = (cell_size * 5 + padding) * len(grids_info) + padding
    screen_height = cell_size * 5 + padding * 2

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Milestone 1: BFS Pathfinding on Multiple Grids")

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))
        draw_grids(screen, grids_info, cell_size, padding)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    main()
