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

        # Step 1: Draw grid title
        title = f"{level} - {algo_name}"
        screen.blit(font.render(title, True, (0, 0, 0)), (x_offset, y_offset - 25))

        # Step 2: Draw each cell (white or black)
        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(x_offset + c * cell_size, y_offset + r * cell_size, cell_size, cell_size)
                color = (220, 220, 220) if grid[r][c] == 0 else (60, 60, 60)
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)

        # Step 3: Draw path (in green)
        for node in path:
            rect = pygame.Rect(x_offset + node[1] * cell_size, y_offset + node[0] * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0, 200, 0), rect)

def main():
    pygame.init()

    # Step 1: Load terrain grid data from file
    terrain_file = os.path.join("data", "terrain_data.json")
    terrain_raw = load_terrain_data(terrain_file)

    # Step 2: Use large grid coordinates for testing real-time performance
    start = (0, 0)
    goal = (9, 9)

    grids_info = []

    for level, grid_list in terrain_raw.items():
        # Step 3: Convert grid to NumPy array
        # Write your code here
        

        # Step 4: Generate cost map using NumPy
        # Write your code here
        # define cost_map here

        # Step 5: Convert grid back to list for compatibility with algorithms
        # write your code here
        # grid = some kind of list

        # Step 6: Run BFS, DFS, Dijkstra
        bfs_path = bfs(grid, start, goal)
        dfs_path = dfs(grid, start, goal)
        dijkstra_path, _ = dijkstra(grid, start, goal, cost_map)

        # Step 7: Add all to visualization queue
        grids_info.append((level, "BFS", grid, bfs_path))
        grids_info.append((level, "DFS", grid, dfs_path))
        grids_info.append((level, "Dijkstra", grid, dijkstra_path))

    # Step 8: Initialize Pygame screen with size adjusted for large grids
    cell_size = 20
    padding = 20
    screen_width = (cell_size * 10 + padding) * 3 + padding
    screen_height = (cell_size * 10 + padding * 3) * 3
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Milestone 4: Optimized Real-Time Pathfinding")

    # Step 9: Main render loop
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
