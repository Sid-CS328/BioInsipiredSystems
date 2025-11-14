# Lab 2

# Particle Swarm Optimization for Function Optimization
# Application: Traffic Light Optimization using PSO
# Goal: Minimize total vehicle delay at a 4-way intersection


import numpy as np
import random


arrival_NS = 35
arrival_EW = 20

CYCLE_TIME = 120

def traffic_delay(times):
    g_ns, g_ew = times

    g_ns = max(g_ns, 1)
    g_ew = max(g_ew, 1)

    delay_NS = arrival_NS / g_ns
    delay_EW = arrival_EW / g_ew

    return delay_NS + delay_EW

NUM_PARTICLES = 20
ITERATIONS = 50

w_inertia = 0.7
c1 = 1.4
c2 = 1.4

particles = []
velocities = []

for _ in range(NUM_PARTICLES):
    g_ns = random.uniform(20, 100)
    g_ew = CYCLE_TIME - g_ns
    particles.append(np.array([g_ns, g_ew]))

    velocities.append(np.random.uniform(-5, 5, 2))

def fitness(x):
    return -traffic_delay(x)

p_best_positions = particles.copy()
p_best_scores = [fitness(p) for p in particles]

g_best_position = p_best_positions[np.argmax(p_best_scores)]

for it in range(ITERATIONS):
    for i in range(NUM_PARTICLES):
        r1, r2 = random.random(), random.random()

        velocities[i] = (
            w_inertia * velocities[i]
            + c1 * r1 * (p_best_positions[i] - particles[i])
            + c2 * r2 * (g_best_position - particles[i])
        )

        particles[i] = particles[i] + velocities[i]

        g_ns = particles[i][0]
        g_ns = max(5, min(g_ns, 115))
        g_ew = CYCLE_TIME - g_ns

        particles[i] = np.array([g_ns, g_ew])

        score = fitness(particles[i])

        if score > p_best_scores[i]:
            p_best_scores[i] = score
            p_best_positions[i] = particles[i]

    g_best_position = p_best_positions[np.argmax(p_best_scores)]

    print(f"Iteration {it+1}: Best Delay = {-max(p_best_scores):.4f}")

best_ns, best_ew = g_best_position
best_delay = -max(p_best_scores)

print("\nOptimized Traffic Signal Timings (PSO):")
print(f"North-South Green Time: {best_ns:.2f} seconds")
print(f"East-West Green Time:   {best_ew:.2f} seconds")
print(f"Total Vehicle Delay Score: {best_delay:.4f}")
