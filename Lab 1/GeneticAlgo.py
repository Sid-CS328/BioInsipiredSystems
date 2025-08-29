#Problem 1: Genetic Algorithm for Optimization Problems

import random


def fitness_function(x):
    return x ** 3


POPULATION_SIZE = 50
GENES_LENGTH = 10
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
GENERATIONS = 50


def random_individual():
    return ''.join(random.choice('01') for _ in range(GENES_LENGTH))

def decode(individual):
    decimal = int(individual, 2)
    return 10 * decimal / (2**GENES_LENGTH - 1)

def evaluate_population(population):
    return [(ind, fitness_function(decode(ind))) for ind in population]


def select(population_fitness):
    total_fitness = sum(f for _, f in population_fitness)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in population_fitness:
        current += fitness
        if current > pick:
            return individual
    return population_fitness[-1][0] 


def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENES_LENGTH - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2


def mutate(individual):
    return ''.join(
        bit if random.random() > MUTATION_RATE else ('1' if bit == '0' else '0')
        for bit in individual
    )


def genetic_algorithm():
    population = [random_individual() for _ in range(POPULATION_SIZE)]
    best_individual = None
    best_fitness = float('-inf')

    for generation in range(GENERATIONS):
        population_fitness = evaluate_population(population)
        population_fitness.sort(key=lambda x: x[1], reverse=True)
        
 
        if population_fitness[0][1] > best_fitness:
            best_fitness = population_fitness[0][1]
            best_individual = population_fitness[0][0]
        
        print(f"Generation {generation}: Best fitness = {best_fitness:.5f}")


        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = select(population_fitness)
            parent2 = select(population_fitness)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

        population = new_population[:POPULATION_SIZE]


    x = decode(best_individual)
    print(f"\nBest solution: x = {x:.5f}, f(x) = {fitness_function(x):.5f}")

genetic_algorithm()
