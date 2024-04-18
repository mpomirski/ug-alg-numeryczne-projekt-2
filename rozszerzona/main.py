import numpy as np


def build_matrix(n, m, alleys, exits, sewage_manhole, starting_points):
    matrix_size = n + 1  # Rozmiar macierzy (liczba skrzyżowań + 1)

    # Inicjalizacja macierzy zerami
    matrix = np.zeros((matrix_size, matrix_size))

    # Wypełnianie macierzy
    for alley in alleys:
        i, j, dij = alley
        matrix[i - 1][j - 1] = matrix[j - 1][i - 1] = -1 / dij

    for exit_point in exits:
        matrix[exit_point - 1][exit_point - 1] = 1

    matrix[sewage_manhole - 1][sewage_manhole - 1] = 0  # OSK

    # Wypełnienie wektora wyrazów wolnych
    b = np.zeros(matrix_size)
    for starting_point in starting_points:
        b[starting_point - 1] = 1

    return matrix, b


def solve_gauss(matrix, b):
    """
    Function to solve the system of equations using Gaussian elimination without pivoting.

    Args:
        matrix (numpy.ndarray): The system of equations matrix.
        b (numpy.ndarray): The free term vector.

    Returns:
        numpy.ndarray: The solution vector or None if the matrix is singular.
    """

    n = len(b)

    # Forward elimination
    for i in range(n):
        if matrix[i][i] == 0:
            return None  # Matrix is singular

        for j in range(i + 1, n):
            ratio = matrix[j][i] / matrix[i][i]
            for k in range(n):
                matrix[j][k] -= ratio * matrix[i][k]
                b[j] -= ratio * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] / matrix[i][i]
        for j in range(i - 1, -1, -1):
            b[j] -= matrix[j][i] * x[i]

    return x


def monte_carlo_simulation(matrix, b, num_trials):
    """
    Function to perform Monte Carlo simulation to verify the solution.

    Args:
        matrix (numpy.ndarray): The system of equations matrix.
        b (numpy.ndarray): The free term vector.
        num_trials (int): Number of trials to simulate.

    Returns:
        float: The estimated probability of safe return home.
    """

    num_safe_returns = 0
    n = len(b)

    for _ in range(num_trials):
        x = np.random.rand(n)  # Random input
        if np.all(np.dot(matrix, x) - b == 0):  # Check if x is a solution
            num_safe_returns += 1

    return num_safe_returns / num_trials


def main():
    """
    Main function to run the program.
    """

    # Read input data
    n, m = map(int, input().split())
    alleys = [list(map(int, input().split())) for _ in range(m)]
    sewage_manhole = int(input())
    exits = list(map(int, input().split()))
    starting_points = list(map(int, input().split()))
    matrix, b = build_matrix(n, m, alleys, exits, sewage_manhole, starting_points)
    resolve = solve_gauss(matrix, b)
    matrix, b = build_matrix(n, m, alleys, exits, sewage_manhole, starting_points)

    # Enhanced Matrix and Vector Printing
    print("System of Equations Matrix:")  # Added a clear label
    for row in matrix:
        print("  ", row)  # Indentation and spacing for better view

    print("\nFree Term Vector:")
    print("  ", b)

    resolve = solve_gauss(matrix, b)

    print("\nSolution Vector:")
    if resolve is not None:
        print("  ", resolve)
    else:
        print("  Matrix is singular.")

    print("\nMonte Carlo Simulation Result:")
    print("  ", monte_carlo_simulation(matrix, b, 10000))


if __name__ == "__main__":
    main()
