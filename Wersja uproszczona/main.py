# Michał Pomirski 15.03.2024
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Random walk matrix
probability_matrix = np.matrix([
    [1,     0,      0,      0,      0,      0],
    [-0.5,  1,      -0.5,   0,      0,      0],
    [0,     -0.5,   1,      -0.5,   0,      0],
    [0,     0,      -0.5,   1,      -0.5,   0],
    [0,     0,      0,      -0.5,   1,      -0.5],
    [0,     0,      0,      0,      0,      1],
])

y = np.array([0, 0, 0, 0, 0, 1]) # p5, p4, p3, p2, p1, p0

result = np.linalg.solve(probability_matrix, y)
print('The result the random walk starting from p3 will end in p0:')
print(f'{result[2]:.2f}')

def random_walk(n):
    position = 3
    for _ in range(n):
        position += np.random.choice([-1, 1], p=[0.5, 0.5])
    return position

# Monte Carlo simulation
successes = 0
trials = 100000
batches = 10
batch_size = trials // batches
pool = mp.Pool(mp.cpu_count())
results = [pool.apply_async(random_walk, args=(batch_size,)) for _ in range(batches)]
pool.close()
pool.join()
successes = np.sum(results == 0)
print('The result of the Monte Carlo simulation:')
print(f'{successes / trials:.2f}')


