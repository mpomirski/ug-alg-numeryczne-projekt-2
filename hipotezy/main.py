import sys
sys.path.append('..')

from rozszerzona.solve_matrix import gauss, gauss_partial_pivoting, gauss_seidel
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

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
                                if matrix[i][j] == 0 and non_zero < non_zero_elements:
                                    matrix[i][j] = 1

                    result_gauss = gauss(matrix, vector)
                    result_gauss_partial_pivoting = gauss_partial_pivoting(matrix, vector)
                    result_numpy = np.linalg.solve(matrix, vector)

                    error_gauss = distance.cdist([result_gauss], [result_numpy], 'euclidean')
                    error_gauss_partial_pivoting = distance.cdist([result_gauss_partial_pivoting], [result_numpy], 'euclidean')

                    errors_gauss = np.append(errors_gauss, error_gauss)
                    errors_gauss_partial_pivoting = np.append(errors_gauss_partial_pivoting, error_gauss_partial_pivoting)

                    tests += 1
                except np.linalg.LinAlgError:
                    continue
            # error_gauss = np.nan_to_num(error_gauss)
            # error_gauss_partial_pivoting = np.nan_to_num(error_gauss_partial_pivoting)

            errors = errors_gauss - errors_gauss_partial_pivoting
            errors = np.nan_to_num(errors)
            errors_sizes = np.append(errors_sizes, np.average(errors))

        plt.subplot(3, 3, non_zero_elements+1)
        plt.plot(sizes, errors_sizes, 'r')
        plt.yscale('log')
        plt.xlabel("Size of matrix")
        plt.title(f"{non_zero_elements}")
        plt.xticks(sizes)
        plt.axhline(0, color='black', lw=1)

    plt.suptitle(f"Difference between the errors of Gauss and Gauss with partial pivoting methods for different sizes of matrix and non-zero elements\n\
                 Number of test cases: {number_of_test_cases}")
    plt.savefig("hypothesis_1.png")
    plt.show()




if __name__ == '__main__':
    test_hypothesis_1()
