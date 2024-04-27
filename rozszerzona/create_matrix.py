import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
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


def create_adjacency_list(graph):
    adj_list = {
        node: list(sorted(graph.neighbors(node), key=lambda x: str(x)))
        for node in graph.nodes()
    }
    return adj_list


def create_transition_matrix(adjacency_list):
    n = len(adjacency_list)
    matrix = np.zeros((n, n))
    for i, (node, neighbours) in enumerate(
        sorted(adjacency_list.items(), key=lambda x: str(x))
    ):
        for neighbour in neighbours:
            matrix[
                i, sorted(adjacency_list.keys(), key=lambda x: str(x)).index(neighbour)
            ] = 1 / len(neighbours)
    return matrix


def generate_equations(transition_matrix):
    n = len(transition_matrix)
    equations = np.eye(n) - transition_matrix
    return equations


def draw_graph(graph, filename="graph.png"):
    nx.draw(graph, with_labels=True, font_weight="bold")
    plt.savefig(filename)


def prepare_data(equations, manhole, park_exit):
    n = len(equations)
    vector = np.zeros(n)
    vector[park_exit - 1] = 1
    equations[manhole - 1] = np.zeros(n)
    equations[park_exit - 1] = np.zeros(n)
    equations[park_exit - 1, park_exit - 1] = 1
    return equations, vector


def main():
    input_data = read_file("data.txt")
    graph = build_graph(input_data)
    draw_graph(graph, "first.png")

    adjacency_list = create_adjacency_list(graph)
    trans_matrix = create_transition_matrix(adjacency_list)
    equations = generate_equations(trans_matrix)

    manhole, park_exit = 3, 1
    matrix, vector = prepare_data(equations, manhole, park_exit)
    test_solve = gauss(matrix, vector)

    print(adjacency_list)
    print(matrix)
    print(vector)
    print(test_solve)


if __name__ == "__main__":
    main()
