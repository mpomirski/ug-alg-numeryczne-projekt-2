import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


def build_park(input_matrix):
    G = nx.Graph()
    id_map = {}
    node_count = 0
    for intersection1, intersection2, steps in input_matrix:
        if intersection1 not in id_map:
            id_map[intersection1] = node_count
            node_count += 1
        if intersection2 not in id_map:
            id_map[intersection2] = node_count
            node_count += 1

        prev_node = id_map[intersection1]
        for step in range(1, steps + 1):
            step_node = node_count
            G.add_node(step_node)
            G.add_edge(prev_node, step_node)
            prev_node = step_node
            node_count += 1
        G.add_edge(prev_node, id_map[intersection2])

    return G, id_map


def adjacency_matrix(G):
    nodes = sorted(G.nodes())
    n = len(nodes)
    adj_matrix = np.zeros((n, n), dtype=int)
    for i, node in enumerate(nodes):
        for neighbor in G.neighbors(node):
            adj_matrix[i][nodes.index(neighbor)] = 1
    return adj_matrix, nodes


def generate_transition_matrix(n, adjacency_list):
    transition_matrix = np.zeros((n, n))
    for i in range(n):
        neighbors = np.where(adjacency_list[i] == 1)[0]
        num_neighbors = len(neighbors)
        for neighbor in neighbors:
            transition_matrix[i][neighbor] = 1 / num_neighbors
    return transition_matrix


def generate_equations(transition_matrix, osk, wyjscie, n):
    A = np.eye(n)
    b = np.zeros(n)
    for i in range(n):
        if i == osk:
            A[i][i] = 1
            b[i] = 0
        elif i == wyjscie:
            A[i][i] = 1
            b[i] = 1
        else:
            A[i][i] = 1
            A[i] -= transition_matrix[i]
    return A, b


def monte_carlo(graph, start, target, pit, num_iterations=10000):
    num_target_reached = 0
    for _ in range(num_iterations):
        current_node = start
        while True:
            neighbors = list(graph.adj[current_node])
            next_node = random.choice(neighbors)
            if next_node == target:
                num_target_reached += 1
                break
            elif next_node == pit:
                break
            current_node = next_node
    return num_target_reached / num_iterations


def main():
    input_matrix = [(1, 2, 3), (2, 3, 2)]  # Example input data
    start = 2  # Start node ID
    osk = 3  # OSK node ID
    wyjscie = 1  # Exit node ID
    num_iterations = 10000  # Number of Monte Carlo simulations

    G, id_map = build_park(input_matrix)
    adj_matrix, nodes = adjacency_matrix(G)
    transition_matrix = generate_transition_matrix(len(nodes), adj_matrix)
    A, b = generate_equations(
        transition_matrix, id_map[osk], id_map[wyjscie], len(nodes)
    )
    probabilities = np.linalg.solve(A, b)
    mc_probability = monte_carlo(
        G, id_map[start], id_map[wyjscie], id_map[osk], num_iterations
    )
    print(A)
    print("Probabilities by Gaussian Elimination:", probabilities)
    print("Monte Carlo Probability:", mc_probability)


if __name__ == "__main__":
    main()
