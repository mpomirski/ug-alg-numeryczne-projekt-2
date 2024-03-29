import numpy as np


def gauss(A, b):
    n = len(A)

    for i in range(n):
        for j in range(i + 1, n):
            ratio = A[j, i] / A[i, i]

            for k in range(n):
                A[j, k] -= ratio * A[i, k]

            b[j] -= ratio * b[i]

    x = np.zeros(n)

    x[n - 1] = b[n - 1] / A[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        sum_ = 0
        for j in range(i + 1, n):
            sum_ += A[i, j] * x[j]
        x[i] = (b[i] - sum_) / A[i, i]

    return np.matrix(x).T


A = np.matrix("1,0,0,0,0,0;-0.5,1,-0.5,0,0,0;0,-0.5,1,-0.5,0,0;0,0,-0.5,1,-0.5,0;0,0,0,-0.5,1,-0.5;0,0,0,0,0,1",
              dtype=np.float64)
b = np.matrix("1;0;0;0;0;0", dtype=np.float64)

x = gauss(A, b)

print(f"Rozwiązanie układu równań:\n{x}")
