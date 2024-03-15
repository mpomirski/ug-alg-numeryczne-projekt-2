# Michał Pomirski 15.03.2024
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import time
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
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    position = 3
    successes = 0
    for _ in range(n):
        while position not in [0, 5]:
            position += np.random.choice([-1, 1], p=[0.5, 0.5])
        if position == 0:
            successes += 1
        position = 3
    return successes

#TODO: Monte Carlo simulation
successes = 0

trials = 100000
batches = 10
batch_size = trials // batches
pool = mp.Pool(mp.cpu_count())
results = [pool.apply_async(random_walk, args=(batch_size,)) for _ in range(batches)]
pool.close()
pool.join()
successes = sum(result.get() for result in results)
print('The result of the Monte Carlo simulation:')
print(f'{successes / trials:.2f}')


