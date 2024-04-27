import numpy as np


def gauss(matrix, vector):
    n = len(matrix)
    A = np.array(matrix, dtype=float)
    b = np.array(vector, dtype=float)

    for i in range(n):
        if np.abs(A[i, i]) == 0:
            for k in range(i + 1, n):
                if np.abs(A[k, i]) > np.abs(A[i, i]):
                    A[[i, k]], A[[k, i]] = A[[k, i]], A[[i, k]]
                    b[[i, k]], b[[k, i]] = b[[k, i]], b[[i, k]]
                    break

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]

    result = np.zeros(n)
    for i in range(n - 1, -1, -1):
        result[i] = (b[i] - np.dot(A[i, i + 1 :], result[i + 1 :])) / A[i, i]
    return result


def gauss_partial_pivoting(A, b):
    n = len(A)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    for i in range(n):
        max_row_index = np.argmax(np.abs(A[i:n, i])) + i
        if i != max_row_index:
            A[[i, max_row_index]], A[[max_row_index, i]] = (
                A[[max_row_index, i]],
                A[[i, max_row_index]],
            )
            b[[i, max_row_index]], b[[max_row_index, i]] = (
                b[[max_row_index, i]],
                b[[i, max_row_index]],
            )

        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    result = np.zeros(n)
    for k in range(n - 1, -1, -1):
        result[k] = (b[k] - np.dot(A[k, k + 1 :], result[k + 1 :])) / A[k, k]
    return result


def gauss_seidel(A, b, tolerance=1e-10, max_iterations=10000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(A)
    x = np.zeros(n)
    x_new = np.copy(x)

    for iteration in range(max_iterations):
        for i in range(n):
            sum_ = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1 :], x[i + 1 :])
            x_new[i] = (b[i] - sum_) / A[i, i] if A[i, i] != 0 else x[i]

        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new
        x = np.copy(x_new)
    return x_new


if __name__ == "__main__":
    A1 = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b1 = [8, -11, -3]

    A2 = [[1, 2, 3], [2, 5, 3], [1, 0, 8]]
    b2 = [10, 8, 3]

    A3 = [[10, 1, 2], [2, 10, 3], [3, 4, 20]]
    b3 = [13, 15, 27]

    x1_gauss = gauss(A1, b1)
    x2_gauss = gauss(A2, b2)
    x3_gauss = gauss(A3, b3)
    x1_gauss_partial_pivoting = gauss_partial_pivoting(A1, b1)
    x2_gauss_partial_pivoting = gauss_partial_pivoting(A2, b2)
    x3_gauss_partial_pivoting = gauss_partial_pivoting(A3, b3)
    x1_gauss_seidel = gauss_seidel(A1, b1)
    x2_gauss_seidel = gauss_seidel(A2, b2)
    x3_gauss_seidel = gauss_seidel(A3, b3)

    print("Gauss method:", x1_gauss, x2_gauss, x3_gauss)
    print(
        "Gauss method with partial pivoting:",
        x1_gauss_partial_pivoting,
        x2_gauss_partial_pivoting,
        x3_gauss_partial_pivoting,
    )
    print("Gauss-Seidel method:", x1_gauss_seidel, x2_gauss_seidel, x3_gauss_seidel)
