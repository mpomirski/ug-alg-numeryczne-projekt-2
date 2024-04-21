import numpy as np


def build_matrix(n, alleys, exits, sewage_manholes):
    size = n + sum(l - 1 for _, _, l in alleys)
    matrix = np.zeros((size, size))
    free_factors = np.zeros(size)
    current_index = n
    node_mapping = list(range(n)) + [None] * (size - n)

    for start, end, length in alleys:
        start, end = start - 1, end - 1
        if length == 1:
            matrix[start, end] = matrix[end, start] = -1
        else:
            intermediates = range(current_index, current_index + length - 1)
            current_index += length - 1
            path = [start] + list(intermediates) + [end]
            for i in range(len(path) - 1):
                matrix[path[i], path[i + 1]] = matrix[path[i + 1], path[i]] = -1

    for i in range(size):
        row_sum = -matrix[i].sum()
        if row_sum != 0:
            matrix[i] /= row_sum
        matrix[i, i] = 1

    for _exit in exits:
        matrix[_exit - 1] = np.zeros(size)
        matrix[_exit - 1, _exit - 1] = 1
        free_factors[_exit - 1] = 1
    for osk in sewage_manholes:
        matrix[osk - 1] = np.zeros(size)
        matrix[osk - 1, osk - 1] = 1

    return matrix


def solve_matrix(matrix):
    n = len(matrix)
    b = np.zeros(n)
    b[[_exit - 1 for _exit in exits]] = 1
    return np.linalg.solve(matrix, b)


def monte_carlo_simulation(matrix, num_trials, start_node):
    n = len(matrix)
    success_count = 0
    for _ in range(num_trials):
        current = start_node
        while True:
            probabilities = matrix[current]
            next_node = np.random.choice(n, p=probabilities)
            if next_node == current:
                if current in [e - 1 for e in exits]:
                    success_count += 1
                break
            current = next_node
    return success_count / num_trials


n = 3
alleys = [(1, 2, 3), (2, 3, 2)]
exits = [1]
sewage_manholes = [3]
start_node = 1 - 1

matrix = build_matrix(n, alleys, exits, sewage_manholes)
probabilities = solve_matrix(matrix)
monte_carlo_prob = monte_carlo_simulation(matrix, 10000, start_node)

print("Matrix:\n", matrix)
print("Probabilities:", probabilities)
print("Monte Carlo Probability:", monte_carlo_prob)
