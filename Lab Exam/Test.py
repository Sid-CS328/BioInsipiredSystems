# Graph Colouring Problem
# Algorithm Used: Genetic Algorithm




import random
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adj_list = {i: [] for i in range(num_nodes)}

    def add_edge(self, u, v):
        if u < 0 or u >= self.num_nodes or v < 0 or v >= self.num_nodes:
            raise ValueError("Node index out of bounds")
        if v not in self.adj_list[u]:
            self.adj_list[u].append(v)
        if u not in self.adj_list[v]:
            self.adj_list[v].append(u)

    def get_neighbors(self, node):
        return self.adj_list[node]

def create_individual(graph, max_colors):
    return [random.randint(0, max_colors - 1) for _ in range(graph.num_nodes)]

def fitness_function(individual, graph):
    conflicts = 0
    for i in range(graph.num_nodes):
        for neighbor in graph.get_neighbors(i):
            if i < neighbor:
                if individual[i] == individual[neighbor]:
                    conflicts += 1

    num_unique_colors = len(set(individual))
    penalty = conflicts * (graph.num_nodes * graph.num_nodes)
    penalty += num_unique_colors

    return -penalty

def selection(population, fitness_scores, num_parents):
    parents = []
    population_size = len(population)
    tournament_size = 3

    for _ in range(num_parents):
        tournament_contenders = random.sample(list(range(population_size)), min(tournament_size, population_size))
        best_contender_index = tournament_contenders[0]
        for i in tournament_contenders:
            if fitness_scores[i] > fitness_scores[best_contender_index]:
                best_contender_index = i
        parents.append(population[best_contender_index])
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)

    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

    return offspring1, offspring2

def mutate(individual, graph, max_colors, mutation_rate):
    for i in range(graph.num_nodes):
        if random.random() < mutation_rate:
            individual[i] = random.randint(0, max_colors - 1)
    return individual

def genetic_algorithm(graph, population_size, max_colors, generations, mutation_rate, num_parents_to_select):
    population = [create_individual(graph, max_colors) for _ in range(population_size)]

    best_individual = None
    best_fitness = float('-inf')

    for generation in range(generations):
        fitness_scores = [fitness_function(ind, graph) for ind in population]

        current_best_idx = fitness_scores.index(max(fitness_scores))
        current_best_individual = population[current_best_idx]
        current_best_fitness = fitness_scores[current_best_idx]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = list(current_best_individual)

        print(f"Generation {generation+1}/{generations}, Best Fitness: {best_fitness}")

        if best_fitness >= -(graph.num_nodes + 1):
            temp_conflicts = 0
            if best_individual:
                for i in range(graph.num_nodes):
                    for neighbor in graph.get_neighbors(i):
                        if i < neighbor:
                            if best_individual[i] == best_individual[neighbor]:
                                temp_conflicts += 1
            if temp_conflicts == 0:
                 print(f"Found conflict-free solution at generation {generation+1}!")
                 break

        parents = selection(population, fitness_scores, num_parents_to_select)

        next_population = []
        if best_individual is not None:
            next_population.append(list(best_individual))

        while len(next_population) < population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            offspring1, offspring2 = crossover(parent1, parent2)

            offspring1 = mutate(offspring1, graph, max_colors, mutation_rate)
            offspring2 = mutate(offspring2, graph, max_colors, mutation_rate)

            next_population.append(offspring1)
            if len(next_population) < population_size:
                next_population.append(offspring2)
        
        population = next_population

    final_conflicts = 0
    if best_individual:
        for i in range(graph.num_nodes):
            for neighbor in graph.get_neighbors(i):
                if i < neighbor:
                    if best_individual[i] == best_individual[neighbor]:
                        final_conflicts += 1
    
    final_unique_colors = len(set(best_individual)) if best_individual else 0

    print("\n--- Genetic Algorithm Finished ---")
    print(f"Best coloring found: {best_individual}")
    print(f"Conflicts: {final_conflicts}")
    print(f"Colors Used: {final_unique_colors}")
    
    return best_individual, final_conflicts, final_unique_colors

print('--- Demonstrating Genetic Algorithm on an Example Graph ---')


num_nodes = 7
example_graph = Graph(num_nodes)

example_graph.add_edge(0, 1)
example_graph.add_edge(0, 2)
example_graph.add_edge(0, 3)
example_graph.add_edge(1, 2)
example_graph.add_edge(1, 4)
example_graph.add_edge(2, 5)
example_graph.add_edge(3, 6)
example_graph.add_edge(4, 5)
example_graph.add_edge(4, 6)
example_graph.add_edge(5, 6)

print(f"Example Graph created with {example_graph.num_nodes} nodes.")
print("Edges:")
for node, neighbors in example_graph.adj_list.items():
    print(f"  Node {node}: {neighbors}")


population_size = 50
max_colors = 5
generations = 200
mutation_rate = 0.1
num_parents_to_select = 20

print("\nGenetic Algorithm Parameters:")
print(f"  Population Size: {population_size}")
print(f"  Max Colors (initial search space): {max_colors}")
print(f"  Generations: {generations}")
print(f"  Mutation Rate: {mutation_rate}")
print(f"  Parents to Select: {num_parents_to_select}")

best_coloring, conflicts, unique_colors = genetic_algorithm(
    example_graph, 
    population_size, 
    max_colors, 
    generations, 
    mutation_rate, 
    num_parents_to_select
)

print("\n--- Genetic Algorithm Results ---")
print(f"Best Coloring Found: {best_coloring}")
print(f"Number of Conflicts: {conflicts}")
print(f"Number of Unique Colors Used: {unique_colors}")

G = nx.Graph()
G.add_nodes_from(range(example_graph.num_nodes))
for u in range(example_graph.num_nodes):
    for v in example_graph.get_neighbors(u):
        if u < v:
            G.add_edge(u, v)

colors = [plt.get_cmap('tab10')(c % 10) for c in best_coloring]








plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=700, font_size=10, font_weight='bold')
plt.title(f"Graph Coloring using Genetic Algorithm\nConflicts: {conflicts}, Colors Used: {unique_colors}")
plt.show()
