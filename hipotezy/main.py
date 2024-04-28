import sys
import time

sys.path.append("..")

from rozszerzona.solve_matrix import gauss, gauss_partial_pivoting, gauss_seidel
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def is_diagonally_dominant(matrix):
    D = np.diag(np.abs(matrix))
    S = np.sum(np.abs(matrix), axis=1) - D
    return np.all(D > S)


def test_hypothesis_1():
    sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    number_of_test_cases = 1000
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    for non_zero_elements in range(9):
        errors_sizes = np.array([])
        for i, size in enumerate(sizes):
            error_gauss = 0
            error_gauss_partial_pivoting = 0
            errors_gauss = np.array([])
            errors_gauss_partial_pivoting = np.array([])
            tests = 0
            while tests < number_of_test_cases:
                try:
                    matrix = np.random.randint(0, 10, (size, size))
                    vector = np.random.randint(0, 10, size)
                    non_zero = np.count_nonzero(matrix)
                    if non_zero < non_zero_elements:
                        for i in range(size):
                            for j in range(size):
                                if matrix[i][j] == 0 and np.count_nonzero(matrix) < non_zero_elements:
                                    matrix[i][j] = 1

                    result_gauss = gauss(matrix, vector)
                    result_gauss_partial_pivoting = gauss_partial_pivoting(
                        matrix, vector
                    )
                    result_numpy = np.linalg.solve(matrix, vector)

                    error_gauss = distance.cdist(
                        [result_gauss], [result_numpy], "euclidean"
                    )
                    error_gauss_partial_pivoting = distance.cdist(
                        [result_gauss_partial_pivoting], [result_numpy], "euclidean"
                    )

                    errors_gauss = np.append(errors_gauss, error_gauss)
                    errors_gauss_partial_pivoting = np.append(
                        errors_gauss_partial_pivoting, error_gauss_partial_pivoting
                    )

                    tests += 1
                except np.linalg.LinAlgError:
                    continue
            # error_gauss = np.nan_to_num(error_gauss)
            # error_gauss_partial_pivoting = np.nan_to_num(error_gauss_partial_pivoting)

            errors = errors_gauss - errors_gauss_partial_pivoting
            errors = np.nan_to_num(errors)
            errors_sizes = np.append(errors_sizes, np.average(errors))

        plt.subplot(3, 3, non_zero_elements + 1)
        plt.plot(sizes, errors_sizes, "r")
        plt.yscale("log")
        plt.xlabel("Size of matrix")
        plt.title(f"{non_zero_elements}")
        plt.xticks(sizes)
        plt.axhline(0, color="black", lw=1)

    plt.suptitle(
        f"Difference between the errors of Gauss and Gauss with partial pivoting methods for different sizes of matrix and non-zero elements\n\
                 Number of test cases: {number_of_test_cases}"
    )
    plt.savefig("hypothesis_1.png")
    plt.show()


def test_hypothesis_2(matrix, vector):
    try:
        result_gauss_seidel = gauss_seidel(matrix, vector)
        return np.all(np.isfinite(result_gauss_seidel))
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def test_hypothesis_3(matrix, vector):
    start_time = time.time()
    _ = gauss(matrix, vector)
    gauss_time = time.time() - start_time

    start_time = time.time()
    _ = gauss_partial_pivoting(matrix, vector)
    gauss_partial_time = time.time() - start_time

    start_time = time.time()
    _ = gauss_seidel(matrix, vector)
    gauss_seidel_time = time.time() - start_time

    return gauss_time, gauss_partial_time, gauss_seidel_time


def main():
    sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    h2_results = []
    h3_results = []

    for size in sizes:
        matrix = np.random.randint(0, 10, (size, size))
        vector = np.random.randint(0, 10, size)

        for i in range(size):
            matrix[i][i] += size

        h2_convergence = test_hypothesis_2(matrix, vector)
        h2_results.append(h2_convergence)

        gauss_time, gauss_partial_time, gauss_seidel_time = test_hypothesis_3(
            matrix, vector
        )
        h3_results.append((gauss_time, gauss_partial_time, gauss_seidel_time))

        print(f"Size {size}:")
        print(f"  H2 - Convergence: {h2_convergence}")
        print(
            f"  H3 - Times: Gauss: {gauss_time}, Gauss Partial: {gauss_partial_time}, Gauss-Seidel: {gauss_seidel_time}"
        )

    plt.figure(figsize=(10, 5))
    gauss_times, gauss_partial_times, gauss_seidel_times = zip(*h3_results)
    plt.plot(sizes, gauss_times, label="Gauss", marker="o")
    plt.plot(sizes, gauss_partial_times, label="Gauss Partial", marker="o")
    plt.plot(sizes, gauss_seidel_times, label="Gauss-Seidel", marker="o")
    plt.xlabel("Size of Matrix")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig("hypothesis_3.png")

    h2_interpretation = (
        "Gauss-Seidel converges"
        if all(h2_results)
        else "Gauss-Seidel does not always converge"
    )
    print(f"Hypothesis 2 Interpretation: {h2_interpretation}")

    faster_methods = []
    if all(
        gauss_seidel < gauss
        for gauss_seidel, gauss in zip(gauss_seidel_times, gauss_times)
    ):
        faster_methods.append("Gauss-Seidel is faster than Gauss")
    if all(
        gauss_seidel < gauss_partial
        for gauss_seidel, gauss_partial in zip(gauss_seidel_times, gauss_partial_times)
    ):
        faster_methods.append("Gauss-Seidel is faster than Gauss Partial")
    h3_interpretation = (
        " and ".join(faster_methods)
        if faster_methods
        else "Gauss-Seidel is not consistently faster"
    )
    print(f"Hypothesis 3 Interpretation: {h3_interpretation}")


if __name__ == "__main__":
    main()
