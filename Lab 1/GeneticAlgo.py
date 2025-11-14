# Lab 1

# Genetic Algorithm for Optimization Problems
# Application: Portfolio Optimization (with Diversification) Using Genetic Algorithm
# Goal: Maximize (Return - Risk_penalty * Risk)


import random
import numpy as np


expected_returns = np.array([0.12, 0.10, 0.15, 0.09])

cov_matrix = np.array([
    [0.006, -0.002, 0.004, 0.000],
    [-0.002, 0.005, -0.001, 0.002],
    [0.004, -0.001, 0.010, 0.003],
    [0.000, 0.002, 0.003, 0.007]
])

risk_penalty = 0.5
diversification_weight = 0.05

POP_SIZE = 30
GENERATIONS = 60
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
NUM_ASSETS = 4

def create_individual():
    weights = np.random.rand(NUM_ASSETS)
    return weights / np.sum(weights)

def fitness(weights):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.dot(weights.T, np.dot(cov_matrix, weights))

    diversification = -np.sum(weights**2)

    return portfolio_return - risk_penalty * portfolio_risk + \
           diversification_weight * diversification

def select_parent(population):
    a, b = random.sample(population, 2)
    return a if fitness(a) > fitness(b) else b

def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, NUM_ASSETS - 1)
        child = np.concatenate((p1[:point], p2[point:]))
        return child / np.sum(child)
    return p1

def mutate(individual):
    if random.random() < MUTATION_RATE:
        index = random.randint(0, NUM_ASSETS - 1)
        individual[index] += np.random.uniform(-0.1, 0.1)
        individual = np.maximum(individual, 0)
        return individual / np.sum(individual)
    return individual

population = [create_individual() for _ in range(POP_SIZE)]
best_solution = None

for gen in range(GENERATIONS):
    new_population = []

    for _ in range(POP_SIZE):
        p1 = select_parent(population)
        p2 = select_parent(population)
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)

    population = new_population

    best = max(population, key=fitness)
    if best_solution is None or fitness(best) > fitness(best_solution):
        best_solution = best

    print(f"Generation {gen+1}: Best Fitness = {fitness(best_solution):.5f}")

print("\nOptimal Diversified Portfolio:")
for i, w in enumerate(best_solution):
    print(f"Asset {i+1}: {w*100:.2f}%")

print(f"\nExpected Return: {np.dot(best_solution, expected_returns):.4f}")
print(f"Portfolio Risk: {np.dot(best_solution.T, np.dot(cov_matrix, best_solution)):.4f}")
print(f"Final Fitness: {fitness(best_solution):.4f}")
