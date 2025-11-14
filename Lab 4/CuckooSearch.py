# Lab 4

# Cuckoo Search (CS)
# Application: Drone Trajectory Planning using Cuckoo Search



import numpy as np
import random


START = np.array([0, 0])
GOAL = np.array([10, 10])

obstacles = [
    (4, 4, 2),
    (7, 6, 1.5)
]

NUM_WAYPOINTS = 5
NUM_NESTS = 20
PA = 0.25
ITERATIONS = 60
BOUND_LOW, BOUND_HIGH = 0, 10

def random_path():
    return np.random.uniform(BOUND_LOW, BOUND_HIGH, (NUM_WAYPOINTS, 2))

def collision_penalty(path):
    penalty = 0
    for (cx, cy, r) in obstacles:
        for (x, y) in path:
            if np.linalg.norm([x - cx, y - cy]) < r:
                penalty += 1
    return penalty

def path_length(path):
    points = [START] + list(path) + [GOAL]
    dist = 0
    for i in range(len(points) - 1):
        dist += np.linalg.norm(points[i+1] - points[i])
    return dist

def smoothness(path):
    points = [START] + list(path) + [GOAL]
    angle_penalty = 0
    for i in range(1, len(points)-1):
        v1 = points[i] - points[i-1]
        v2 = points[i+1] - points[i]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
        angle_penalty += (1 - cos_angle)
    return angle_penalty

def fitness(path):
    L = path_length(path)
    C = collision_penalty(path)
    S = smoothness(path)
    return L + 20*C + 0.1*S

def levy_flight(Lambda):
    u = np.random.normal(0, 1)
    v = np.random.normal(0, 1)
    return u / (abs(v)**(1/Lambda))

nests = [random_path() for _ in range(NUM_NESTS)]
best_path = None
best_cost = float("inf")

for it in range(ITERATIONS):
    new_nests = []
    for nest in nests:
        step = levy_flight(1.5)
        new_path = nest + step * np.random.randn(NUM_WAYPOINTS, 2)
        new_path = np.clip(new_path, BOUND_LOW, BOUND_HIGH)
        new_nests.append(new_path)

    for i in range(NUM_NESTS):
        if fitness(new_nests[i]) < fitness(nests[i]):
            nests[i] = new_nests[i]

    worst_k = int(PA * NUM_NESTS)
    worst_indices = np.argsort([fitness(n) for n in nests])[-worst_k:]

    for idx in worst_indices:
        nests[idx] = random_path()

    current_best_idx = np.argmin([fitness(n) for n in nests])
    current_cost = fitness(nests[current_best_idx])

    if current_cost < best_cost:
        best_cost = current_cost
        best_path = nests[current_best_idx]

    print(f"Iteration {it+1}: Best Cost = {best_cost:.4f}")

print("\nOptimal Drone Path Found (Cuckoo Search):")
print("Waypoints:")
print(best_path)
print(f"\nFinal Cost = {best_cost:.4f}")
