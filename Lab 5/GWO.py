# Lab 5

# Grey Wolf Optimizer (GWO)
# Application: Support Vector Machine (SVM) Hyperparameter Optimization using GWO to maximize classification accuracy on the Iris dataset.



import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


iris = datasets.load_iris()
X = iris.data
y = iris.target

def fitness(params):
    C, gamma = params
    C = max(C, 1e-5)
    gamma = max(gamma, 1e-5)
    model = SVC(C=C, gamma=gamma)
    score = cross_val_score(model, X, y, cv=5).mean()
    return -score

def GWO(fitness, dim, n_wolves=10, n_iter=30, lb=None, ub=None):
    if lb is None:
        lb = np.zeros(dim)
    if ub is None:
        ub = np.ones(dim)
    
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))
    
    alpha_pos, alpha_score = None, float("inf")
    beta_pos, beta_score = None, float("inf")
    delta_pos, delta_score = None, float("inf")
    
    for t in range(n_iter):
        a = 2 - t * (2 / n_iter)
        
        for i in range(n_wolves):
            wolves[i] = np.clip(wolves[i], lb, ub)
            
            score = fitness(wolves[i])
            
            if score < alpha_score:
                alpha_score, alpha_pos = score, wolves[i].copy()
            elif score < beta_score:
                beta_score, beta_pos = score, wolves[i].copy()
            elif score < delta_score:
                delta_score, delta_pos = score, wolves[i].copy()
        
        for i in range(n_wolves):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos - wolves[i])
            X1 = alpha_pos - A1 * D_alpha

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_pos - wolves[i])
            X2 = beta_pos - A2 * D_beta

            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_pos - wolves[i])
            X3 = delta_pos - A3 * D_delta

            wolves[i] = (X1 + X2 + X3) / 3
        
        print(f"Iteration {t+1}: Best Fitness = {-alpha_score:.5f}")
    
    return alpha_pos, -alpha_score

lb = [0.01, 0.001]
ub = [100, 1]

best_params, best_score = GWO(fitness, dim=2, n_wolves=15, n_iter=30, lb=lb, ub=ub)

print("\nOptimal SVM Parameters Found:")
print(f"C = {best_params[0]:.5f}, gamma = {best_params[1]:.5f}")
print(f"Cross-Validated Accuracy = {best_score:.5f}")
