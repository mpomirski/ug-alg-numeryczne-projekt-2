import numpy as np

def back_substitution(matrix: np.matrix[int, np.dtype[np.float64]], vector: np.ndarray[int, np.dtype[np.float64]]) -> np.ndarray[int, np.dtype[np.float64]]:
    n: int = len(vector)
    x: np.ndarray[int, np.dtype[np.float64]] = np.zeros(n)
    x[n-1] = vector[n-1] / matrix[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = vector[i]
        for j in range(i+1, n):
            x[i] -= matrix[i, j] * x[j]
        x[i] /= matrix[i, i]
    return x

def gauss_partial_elimination(matrix: np.matrix[int, np.dtype[np.float64]], vector: np.ndarray[int, np.dtype[np.float64]]) -> np.ndarray[int, np.dtype[np.float64]]:
    maxima: np.ndarray[int, np.dtype[np.float64]] = np.max(matrix, axis=0)
    for k, b in enumerate(vector):
        for i in range(k+1, len(vector)):
            if matrix[k, k] == 0:
                matrix[k, k] = maxima[:, k]
            factor: np.float64 = matrix[i, k] / matrix[k, k]
            for j in range(k, len(vector)):
                matrix[i, j] -= factor * matrix[k, j]
            vector[i] -= factor * b
    return back_substitution(matrix, vector)
    

def test_gauss_partial_elimination1() -> None:
    matrix: np.matrix[int, np.dtype[np.float64]] = np.matrix(
        [
        [1.2, 2.6, -0.1, 1.5],
        [4.5, 9.8, -0.4, 5.7],
        [0.1, -0.1, -0.3, -3.5],
        [4.5, -5.2, 4.2, -3.4]
        ],
        dtype=np.float64
        )
    vector: np.ndarray[int, np.dtype[np.float64]] = np.array([13.15, 49.84, -14.08, -46.51], dtype=np.float64)
    result: np.ndarray[int, np.dtype[np.float64]] = gauss_partial_elimination(matrix, vector)
    assert np.allclose(result, np.array([-1.3, 3.2, -2.4, 4.1], dtype=np.float64))

def test_gauss_partial_elimination2() -> None:
    matrix: np.matrix[int, np.dtype[np.float64]] = np.matrix(
    [
    [2, 4, 2, 0],
    [1, 0, -1, 1],
    [0, 1, 3, -1],
    [2, 1, 2, 1]
    ], 
    dtype=np.float64
    )
    vector: np.ndarray[int, np.dtype[np.float64]] = np.array([4, 2, 0, 6], dtype=np.float64)
    result: np.ndarray[int, np.dtype[np.float64]] = gauss_partial_elimination(matrix, vector)
    assert np.allclose(result, np.array([-4, 2, 2, 8], dtype=np.float64))

def main() -> None:
    test_gauss_partial_elimination1()
    test_gauss_partial_elimination2()

if __name__ == "__main__":
    main()