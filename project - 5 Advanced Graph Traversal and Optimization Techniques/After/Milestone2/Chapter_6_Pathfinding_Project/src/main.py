import pygame
import json
import os
from algorithms.bfs import bfs
from algorithms.dfs import dfs

def load_terrain_data(filepath):
    """
    Loads terrain grid data for each difficulty level from a JSON file.
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def draw_grids(screen, grids_info, cell_size=40, padding=40):
    """
    Draws BFS and DFS grids side-by-side in one window.

    Args:
        screen: Pygame screen surface.
        grids_info: List of (level, algorithm, grid, path) tuples.
        cell_size: Pixel size of each cell.
        padding: Spacing between grids.
    """
    font = pygame.font.SysFont(None, 24)

    for idx, (level, algo_name, grid, path) in enumerate(grids_info):
        rows, cols = len(grid), len(grid[0])
        x_offset = padding + (cols * cell_size + padding) * (idx % 3)
        y_offset = padding + (rows * cell_size + 3 * padding) * (idx // 3)

        # Draw label above grid
        title = f"{level} - {algo_name}"
        screen.blit(font.render(title, True, (0, 0, 0)), (x_offset, y_offset - 25))

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
                color = (200, 200, 200) if grid[r][c] == 0 else (50, 50, 50)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        # Draw path
        for node in path:
            rect = pygame.Rect(x_offset + node[1] * cell_size, y_offset + node[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 200, 0), rect)

def main():
    """
    Loads terrain data and visualizes BFS and DFS pathfinding
    results for Easy, Medium, and Hard grids.
    """
    pygame.init()

    terrain_file = os.path.join("../data", "terrain_data.json")
    terrain = load_terrain_data(terrain_file)
    start, goal = (0, 0), (4, 4)

    grids_info = []
    for level, grid in terrain.items():
        bfs_path = bfs(grid, start, goal)
        dfs_path = dfs(grid, start, goal)
        grids_info.append((level, "BFS", grid, bfs_path))
        grids_info.append((level, "DFS", grid, dfs_path))

    # Setup window size
    cell_size = 40
    padding = 40
    screen_width = (cell_size * 5 + padding) * 3 + padding
    screen_height = (cell_size * 5 + padding * 3) * 2
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Milestone 2: BFS vs DFS Pathfinding")

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
