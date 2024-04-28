import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from solve_matrix import gauss


def read_file(file_name):
    data = []
    with open(file_name, "r") as file:
        for line in file:
            parts = list(map(int, line.split()))
            parts[-1] -= 1
            data.append(parts)
    return data


def build_graph(edges):
    graph = nx.Graph()
    for start, end, steps in edges:
        graph.add_node(start)
        graph.add_node(end)
        prev_node = start
        for step in range(1, steps + 1):
            step_node = f"{start}{end}{step}"
            graph.add_node(step_node)
            graph.add_edge(prev_node, step_node)
            prev_node = step_node
        graph.add_edge(prev_node, end)
    return graph


def custom_sort(node):
    if isinstance(node, int):
        return (0, node)
    if isinstance(node, str):
        return (1, int("".join(filter(str.isdigit, node))), node)
    raise ValueError("Node is neither an int nor a string.")


def create_adjacency_list(graph):
    return {
        node: list(sorted(graph.neighbors(node), key=custom_sort))
        for node in sorted(graph.nodes(), key=custom_sort)
    }


def create_transition_matrix(adjacency_list):
    n = len(adjacency_list)
    matrix = np.zeros((n, n))
    nodes = sorted(adjacency_list.keys(), key=lambda x: str(x))
    for i, node in enumerate(nodes):
        neighbours = adjacency_list[node]
        for neighbour in neighbours:
            j = nodes.index(neighbour)
            matrix[i, j] = 1 / len(neighbours)
    return matrix


def generate_equations(transition_matrix):
    return np.eye(len(transition_matrix)) - transition_matrix


def draw_graph(graph, filename="graph.png"):
    nx.draw(graph, with_labels=True, font_weight="bold")
    plt.savefig(filename)
    plt.close()


def prepare_data(equations, manhole, park_exit):
    n = len(equations)
    vector = np.zeros(n)
    vector[park_exit] = 1
    equations[manhole] = np.zeros(n)
    equations[manhole, manhole] = 1
    equations[park_exit] = np.zeros(n)
    equations[park_exit, park_exit] = 1
    return equations, vector


def find_node_indices(graph, exit_label, manhole_label):
    start_node = exit_label if exit_label in graph.nodes() else manhole_label

    queue = deque([start_node])
    visited = set()

    node_indices = {}
    counter = 0

    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)

            if current_node == exit_label or current_node == manhole_label:
                node_indices[current_node] = counter

            counter += 1

            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    queue.append(neighbor)

    exit_index = node_indices.get(exit_label, None)
    manhole_index = node_indices.get(manhole_label, None)

    return exit_index, manhole_index


def main():
    input_data = read_file("data.txt")
    graph = build_graph(input_data)
    draw_graph(graph, "first.png")

    adjacency_list = create_adjacency_list(graph)
    trans_matrix = create_transition_matrix(adjacency_list)
    equations = generate_equations(trans_matrix)
    exit_index, manhole_index = find_node_indices(graph, 1, 3)

    matrix, vector = prepare_data(equations, manhole_index, exit_index)
    test_solve = gauss(matrix, vector)

    print(manhole_index, exit_index)
    print(matrix)
    print(vector)
    print(test_solve)


if __name__ == "__main__":
    main()
