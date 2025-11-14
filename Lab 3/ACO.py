# Lab 3

# Ant Colony Optimization for the Traveling Salesman Problem
# Application: Delivery Route Optimization using ACO (TSP)



import numpy as np
import random
import math


cities = np.array([
    [0, 0],
    [2, 4],
    [3, 1],
    [5, 2],
    [6, 6],
    [8, 3],
    [1, 7],
    [4, 5]
])
num_cities = len(cities)

def euclidean(a, b):
    return np.linalg.norm(a - b)

dist_matrix = np.zeros((num_cities, num_cities))
for i in range(num_cities):
    for j in range(num_cities):
        dist_matrix[i][j] = euclidean(cities[i], cities[j])

num_ants = 20
alpha = 1.0
beta = 5.0
rho = 0.5
Q = 100
iterations = 50

pheromone = np.ones((num_cities, num_cities))
heuristic = 1 / (dist_matrix + np.eye(num_cities))

def construct_route(start_city):
    route = [start_city]
    unvisited = set(range(num_cities))
    unvisited.remove(start_city)

    current = start_city

    while unvisited:
        probabilities = []
        for next_city in unvisited:
            tau = pheromone[current][next_city] ** alpha
            eta = heuristic[current][next_city] ** beta
            probabilities.append(tau * eta)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        next_city = random.choices(list(unvisited), weights=probabilities)[0]
        route.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return route

def route_length(route):
    total = 0
    for i in range(len(route)):
        total += dist_matrix[route[i]][route[(i+1) % num_cities]]
    return total

best_route = None
best_length = math.inf

for it in range(iterations):
    routes = []
    lengths = []

    for ant in range(num_ants):
        start = random.randint(0, num_cities - 1)
        route = construct_route(start)
        L = route_length(route)
        routes.append(route)
        lengths.append(L)

        if L < best_length:
            best_length = L
            best_route = route

    pheromone *= (1 - rho)

    for route, L in zip(routes, lengths):
        for i in range(num_cities):
            a = route[i]
            b = route[(i + 1) % num_cities]
            pheromone[a][b] += Q / L
            pheromone[b][a] += Q / L

    print(f"Iteration {it+1}: Best Length = {best_length:.3f}")

print("\nOptimal Delivery Route (ACO):")
print(" -> ".join(str(city) for city in best_route))
print(f"Shortest Distance: {best_length:.3f}")
