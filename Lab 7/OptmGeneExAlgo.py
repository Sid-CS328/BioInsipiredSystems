# Lab 7

# Optimization via Gene Expression Algorithms
# Application: Feature Selection in Machine Learning



import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


data = load_breast_cancer()
X = data.data
y = data.target
num_features = X.shape[1]

population_size = 20
generations = 30
mutation_rate = 0.1
crossover_rate = 0.8
num_genes = num_features

population = np.random.randint(2, size=(population_size, num_genes))

def fitness(individual):
    selected_features = [i for i, gene in enumerate(individual) if gene == 1]
    if not selected_features:
        return 0
    X_subset = X[:, selected_features]
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    score = cross_val_score(clf, X_subset, y, cv=3).mean()
    return score

def select(population, fitness_vals):
    probs = fitness_vals / np.sum(fitness_vals)
    idx = np.random.choice(len(population), size=2, p=probs)
    return population[idx[0]], population[idx[1]]

def crossover(parent1, parent2):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, num_genes-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual):
    for i in range(num_genes):
        if np.random.rand() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual

best_solution = None
best_fitness = 0

for gen in range(generations):
    fitness_vals = np.array([fitness(ind) for ind in population])

    max_idx = np.argmax(fitness_vals)
    if fitness_vals[max_idx] > best_fitness:
        best_fitness = fitness_vals[max_idx]
        best_solution = population[max_idx].copy()
    
    new_population = []
    while len(new_population) < population_size:
        p1, p2 = select(population, fitness_vals)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_population.extend([c1, c2])
    
    population = np.array(new_population[:population_size])
    print(f"Generation {gen+1}: Best Fitness = {best_fitness:.4f}")

selected_features = [i for i, gene in enumerate(best_solution) if gene == 1]
print("\nOptimal Feature Subset Indices:", selected_features)
print("Number of Features Selected:", len(selected_features))
print("Best Cross-Validated Accuracy:", best_fitness)
