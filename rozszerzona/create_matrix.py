import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def read_file(file_name):
    data = []
    with open(file_name, "r") as file:
        for line in file:
            parts = list(map(int, line.split()))
            parts[-1] -= 1
            data.append
    return data


def build_graph(edges):
    graph = nx.Graph()
    for start, end, steps in edges:
        graph.add_node(start)
        graph.add_node(end)
        prev_node = start
        for step in range(1, steps + 1):
            step_node = f"{start}-{end}-{step}"
            graph.add_node(step_node)
            graph.add_edge(prev_node, step_node)
            prev_node = step_node
        graph.add_edge(prev_node, end)
    return graph


def create_adjacency_list(graph: nx.Graph):
    adj_list = {node: list(sorted(graph.neighbors(node))) for node in graph.nodes()}
    return adj_list


def create_transition_matrix(adjacency_list):
    n = len(adjacency_list)
    matrix = np.zeros((n, n))
    for i, neighbours in adjacency_list.items():
        for neighbour in neighbours:
            matrix[i, neighbour] = 1 / len(neighbours)
    return matrix


def generate_equations(transition_matrix):
    n = len(transition_matrix)
    equations = np.eye(n) - transition_matrix
    return equations


def draw_graph(graph, filename="graph.png"):
    nx.draw(graph, with_labels=True, font_weight="bold")
    plt.savefig(filename)


def main():
    pass


if __name__ == "__main__":
    main()
