# Lab 6

# Parallel Cellular Algorithms and Programs
# Application: Multi-Core Task Scheduling Optimization



import numpy as np


N = 10
M = 3
tasks = np.random.randint(1, 20, size=N)

grid_size = (5, 5)
iterations = 50

cells = np.random.randint(0, M, size=(grid_size[0], grid_size[1], N))

def fitness(cell):
    core_times = np.zeros(M)
    for task_idx, core in enumerate(cell):
        core_times[core] += tasks[task_idx]
    return np.max(core_times)


def get_neighbors(grid, i, j):
    neighbors = []
    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        ni, nj = i+di, j+dj
        if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
            neighbors.append(grid[ni, nj])
    return neighbors


best_solution = None
best_fitness = float('inf')

for it in range(iterations):
    new_cells = cells.copy()
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            neighbors = get_neighbors(cells, i, j)
            neighbor_fitness = [fitness(n) for n in neighbors]
            if neighbors and min(neighbor_fitness) < fitness(cells[i, j]):
                best_neighbor = neighbors[np.argmin(neighbor_fitness)]
                new_cells[i, j] = best_neighbor.copy()
                if np.random.rand() < 0.2:
                    task_to_change = np.random.randint(0, N)
                    new_cells[i, j, task_to_change] = np.random.randint(0, M)
    cells = new_cells

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            f = fitness(cells[i, j])
            if f < best_fitness:
                best_fitness = f
                best_solution = cells[i, j].copy()

    print(f"Iteration {it+1}: Best Makespan = {best_fitness}")

print("\nOptimal Task Assignment:")
for task_idx, core in enumerate(best_solution):
    print(f"Task {task_idx+1} -> Core {core+1}")
print(f"Minimum Makespan: {best_fitness}")
