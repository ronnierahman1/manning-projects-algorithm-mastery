import pygame
import json
import os
import numpy as np

from algorithms.bfs import bfs
from algorithms.dfs import dfs
from algorithms.dijkstra import dijkstra
from optimization.optimizer import convert_grid_to_numpy, generate_cost_map_numpy

def load_terrain_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def draw_grids(screen, grids_info, cell_size=20, padding=20):
    font = pygame.font.SysFont(None, 22)

    for idx, (level, algo_name, grid, path) in enumerate(grids_info):
        rows, cols = len(grid), len(grid[0])
        x_offset = padding + (cols * cell_size + padding) * (idx % 3)
        y_offset = padding + (rows * cell_size + 3 * padding) * (idx // 3)

        title = f"{level} - {algo_name}"
        screen.blit(font.render(title, True, (0, 0, 0)), (x_offset, y_offset - 25))

        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
                color = (220, 220, 220) if grid[r][c] == 0 else (60, 60, 60)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        for node in path:
            rect = pygame.Rect(x_offset + node[1] * cell_size, y_offset + node[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 200, 0), rect)

def main():
    pygame.init()

    terrain_file = os.path.join("../data", "terrain_data.json")
    terrain_raw = load_terrain_data(terrain_file)
    start, goal = (0, 0), (9, 9)  # Larger grids now supported

    grids_info = []

    for level, grid_list in terrain_raw.items():
        # Convert to NumPy for performance
        grid_np = convert_grid_to_numpy(grid_list)
        cost_map = generate_cost_map_numpy(grid_np)

        # Convert back to list-of-lists for compatibility with algorithms
        grid = grid_np.tolist()

        # Run all algorithms
        bfs_path = bfs(grid, start, goal)
        dfs_path = dfs(grid, start, goal)
        dijkstra_path, _ = dijkstra(grid, start, goal, cost_map)

        grids_info.append((level, "BFS", grid, bfs_path))
        grids_info.append((level, "DFS", grid, dfs_path))
        grids_info.append((level, "Dijkstra", grid, dijkstra_path))

    cell_size = 20
    padding = 20
    screen_width = (cell_size * 10 + padding) * 3 + padding
    screen_height = (cell_size * 10 + padding * 3) * 3
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Milestone 4: Optimized Real-Time Pathfinding")

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
