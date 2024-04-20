import numpy as np


def build_graph_and_matrix(n, alleys, exits, sewage_manholes):
    adjacency = {i: [] for i in range(n)}  # Initial adjacency list for base nodes

    # Expanding graph with intermediate nodes
    node_count = n
    for start, end, length in alleys:
        start -= 1
        end -= 1
        current = start
        for _ in range(length - 1):
            new_node = node_count
            adjacency[current].append(new_node)
            adjacency[new_node] = [current]  # Initialize new node adjacency
            current = new_node
            node_count += 1
        adjacency[current].append(end)
        adjacency[end].append(current)

    # Building the transition matrix
    size = node_count  # Total nodes including intermediates
    matrix = np.zeros((size, size))
    for node, neighbors in adjacency.items():
        if node + 1 in exits:  # +1 to correct zero-indexing for input
            matrix[node] = np.zeros(size)
            matrix[node, node] = 1
        elif node + 1 in sewage_manholes:  # Same indexing correction
            matrix[node] = np.zeros(size)
            matrix[node, node] = 1
        else:
            num_neighbors = len(neighbors)
            matrix[node, node] = 1
            for neighbor in neighbors:
                matrix[node, neighbor] = -1 / num_neighbors

    # Setup the free terms vector for exits
    b = np.zeros(size)
    for exit in exits:
        exit_index = exit - 1  # Adjust for zero-indexing
        b[exit_index] = 1

    return matrix, b, size


def solve_gauss(matrix, b):
    n = len(matrix)
    # Gaussian elimination
    for i in range(n):
        pivot = matrix[i, i]
        if pivot == 0:
            continue
        for j in range(i + 1, n):
            factor = matrix[j, i] / pivot
            matrix[j] -= factor * matrix[i]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = (b[i] - np.sum(matrix[i, i + 1 :] * x[i + 1 :])) / matrix[i, i]

    return x


def main():
    n = int(input("Enter the number of nodes: "))
    alleys = [(1, 2, 3), (2, 3, 2)]
    exits = [1]
    sewage_manholes = [3]

    matrix, b, size = build_graph_and_matrix(n, alleys, exits, sewage_manholes)
    probabilities = solve_gauss(matrix, b)

    print("Matrix:\n", matrix)
    print("Free terms vector (b):", b)
    print("Probabilities of reaching an exit from each node:", probabilities)


if __name__ == "__main__":
    main()
