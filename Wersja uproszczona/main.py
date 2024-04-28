# Michał Pomirski 15.03.2024
import numpy as np
import multiprocessing as mp
import os
import time


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


def test_version(trials: int = 10000) -> float:
    probability_matrix = np.matrix([
        [1,     0,      0,      0,      0,      0],
        [-0.5,  1,      -0.5,   0,      0,      0],
        [0,     -0.5,   1,      -0.5,   0,      0],
        [0,     0,      -0.5,   1,      -0.5,   0],
        [0,     0,      0,      -0.5,   1,      -0.5],
        [0,     0,      0,      0,      0,      1],
    ])

    y = np.array([0, 0, 0, 0, 0, 1])  # p5, p4, p3, p2, p1, p0

    result = np.linalg.solve(probability_matrix, y)

    successes = 0
    batches = 10
    batch_size = trials // batches
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(random_walk, args=(batch_size,))
               for _ in range(batches)]
    pool.close()
    pool.join()
    successes = sum(result.get() for result in results)
    return successes / trials


def simple_version() -> np.float64:
    n: int = 2
    s: int = 3
    p: float = -0.5
    good_position: int = 0
    bad_position: int = 5
    probability_matrix: np.ndarray[np.float64, np.dtype] = np.identity(
        n+s+1, dtype=np.float64)
    for i in range(n+s+1):
        if i == bad_position:
            probability_matrix[i][i] = 1
        elif i == good_position:
            probability_matrix[i][i] = 1
        else:
            probability_matrix[i][i-1] = p
            probability_matrix[i][i+1] = p

    res = np.zeros(n+s+1)
    res[good_position] = 1
    res = np.linalg.solve(probability_matrix, res)
    return res[s]


def main() -> None:
    trials: int = 100_000_000
    print(f'Monte Carlo simulation: {test_version(trials):.4f}')
    print(f'Simple version: {simple_version():.4f}')
    print(f'Error: {abs(test_version() - simple_version()):.4f}')
    print(
        f'Error percentage: {abs(test_version() - simple_version()) / simple_version():.4%}')
    print(
        f'Are results close? {np.isclose(test_version(), simple_version(), atol=0.01)}')


if __name__ == '__main__':
    main()
