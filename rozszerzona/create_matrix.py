import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
from solve_matrix import gauss


def load_data(file_path):
    content = []
    with open(file_path, "r") as file:
        for line in file:
            elements = list(map(int, line.split()))
            elements[-1] -= 1
            content.append(elements)
    return content


def construct_graph(edge_list):
    g = nx.Graph()
    for start, end, steps in edge_list:
        g.add_node(start)
        g.add_node(end)
        last_node = start
        for step in range(1, steps + 1):
            step_node = f"{start}{end}{step}"
            g.add_node(step_node)
            g.add_edge(last_node, step_node)
            last_node = step_node
        g.add_edge(last_node, end)
    return g


def sort_nodes(node):
    if isinstance(node, int):
        return (0, node)
    if isinstance(node, str):
        return (1, int("".join(filter(str.isdigit, node))), node)
    raise ValueError("Node is neither an int nor a string.")


def generate_adjacency_list(g):
    return {
        node: list(sorted(g.neighbors(node), key=sort_nodes))
        for node in sorted(g.nodes(), key=sort_nodes)
    }


def generate_transition_matrix(adj_list):
    size = len(adj_list)
    matrix = np.zeros((size, size))
    nodes = sorted(adj_list.keys(), key=lambda x: str(x))
    for i, node in enumerate(nodes):
        neighbours = adj_list[node]
        for neighbour in neighbours:
            j = nodes.index(neighbour)
            matrix[i, j] = 1 / len(neighbours)
    return matrix


def create_equations(trans_matrix):
    return np.eye(len(trans_matrix)) - trans_matrix


def visualize_graph(g, filename="graph.png"):
    nx.draw(g, with_labels=True, font_weight="bold")
    plt.savefig(filename)
    plt.close()


def process_data(equations, manhole_indices, exit_indices):
    size = len(equations)
    vector = np.zeros(size)
    for exit_index in exit_indices:
        vector[exit_index] = 1
        equations[exit_index] = np.zeros(size)
        equations[exit_index, exit_index] = 1

    for manhole_index in manhole_indices:
        equations[manhole_index] = np.zeros(size)
        equations[manhole_index, manhole_index] = 1

    return equations, vector


def get_node_indices(g, exit_labels, manhole_labels):
    visited = set()
    exit_indices = []
    manhole_indices = []
    for node in g.nodes():
        if node in visited:
            continue
        queue = deque([node])
        counter = 0
        while queue:
            current_node = queue.popleft()
            if current_node not in visited:
                visited.add(current_node)
                if current_node in exit_labels:
                    exit_indices.append(counter)
                if current_node in manhole_labels:
                    manhole_indices.append(counter)
                counter += 1
                for neighbor in g.neighbors(current_node):
                    if neighbor not in visited:
                        queue.append(neighbor)

    return exit_indices, manhole_indices


def execute():
    data = load_data("data.txt")
    g = construct_graph(data)
    visualize_graph(g, "first.png")

    adj_list = generate_adjacency_list(g)
    trans_matrix = generate_transition_matrix(adj_list)
    equations = create_equations(trans_matrix)
    exit_index, manhole_index = get_node_indices(g, [2, 4], [1])
    wanderer_index = 3

    matrix, vector = process_data(equations, manhole_index, exit_index)
    solution = gauss(matrix, vector)

    print(manhole_index, exit_index)
    print(matrix)
    print(vector)
    print(solution[wanderer_index - 1])


if __name__ == "__main__":
    execute()
