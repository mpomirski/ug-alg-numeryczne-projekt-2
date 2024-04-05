# Michał Pomirski 15.03.2024
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import time
# Random walk matrix

def test_version() -> None:
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

def simple_version() -> None:
    n: int = 2
    s: int = 3
    p: float = -0.5
    good_position: int = 0
    bad_position: int = 5
    probability_matrix = np.identity(n+s+1)
    for i in range(n+s+1):
        if i == bad_position:
            probability_matrix[i][i] = 1
        elif i == good_position:
            probability_matrix[i][i] = 1
        else:
            probability_matrix[i][i-1] = p
            probability_matrix[i][i+1] = p
            
    print("Probability matrix:")
    print(probability_matrix)
    res = np.zeros(n+s+1)
    res[good_position] = 1
    res = np.linalg.solve(probability_matrix, res)
    print("Result:")
    print(res)
def extended_version() -> None:
    def random_walk(start: Tuple[int, int], paths: List[Tuple[int, ...]], osk: List[int], exits: List[int], traveler: List[int], sinks: List[int]) -> np.ndarray:
        n, m = start
        probability_matrix = np.zeros((m, m))
        for path in paths:
            i, j, length = path
            probability_matrix[i][j] = 1 / length
        for i in exits:
            probability_matrix[i][i] = 1
        for i in osk:
            probability_matrix[i][i] = 1
        
        return probability_matrix

        

    n, m = (int(x) for x in input().split())
    paths: List[tuple[int, ...]] = []
    for _ in range(m):
        paths.append(tuple(int(x) for x in input().split()))
    osk: List[int] = [int(x) for x in input().split()[1:]]
    exits: List[int] = [int(x) for x in input().split()[1:]]
    traveler: List[int] = [int(x) for x in input().split()[1:]]
    sinks: List[int] = [int(x) for x in input().split()[1:]]

    # print(n, m)
    # print(paths)
    # print(osk)
    # print(exits)
    # print(traveler)
    # print(sinks)
    print("Probability matrix:")
    print(random_walk((n, m), paths, osk, exits, traveler, sinks))


def main() -> None:
    extended_version()

if __name__ == '__main__':
    main()



