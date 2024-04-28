from multiprocessing.pool import AsyncResult
import numpy as np
from typing import Any, Self, TypeAlias
import multiprocessing as mp

npintarray: TypeAlias = np.ndarray[np.int32, np.dtype[Any]]
npfloatarray: TypeAlias = np.ndarray[np.float64, np.dtype[Any]]


class MonteCarlo:
    def __init__(self, trials: int = 10000, batches: int = 10, seed: int = 123456789) -> None:
        self.trials: int = trials
        self.batches: int = batches
        self.seed: int = seed
        self.adjacency_matrix: npfloatarray = np.array([], dtype=np.float64)
        self.y: npintarray = np.array([])
        self.start_position: int | None = None

    def set_adjacency_matrix(self, adjacency_matrix: npfloatarray) -> Self:
        self.adjacency_matrix: npfloatarray = adjacency_matrix
        return self

    def set_y(self, y: npintarray) -> Self:
        self.y: npintarray = y
        return self

    def set_start_position(self, start_position: int) -> Self:
        self.start_position = start_position
        return self

    def _build_probability_matrix(self, adjacency_matrix: npfloatarray) -> Any:
        probability_matrix = adjacency_matrix - \
            np.identity(adjacency_matrix.shape[0], dtype=np.float64)
        probability_matrix: npfloatarray = np.negative(probability_matrix)
        for i in range(probability_matrix.shape[0]):
            for j in range(probability_matrix.shape[1]):
                if np.all(probability_matrix[i] == 0):
                    probability_matrix[i][i] = 1
        return probability_matrix

    def _random_walk(self, n: int, probability_matrix: npintarray, start: int, osk: int, y: npintarray) -> int:
        np.random.seed(123456789)
        successes = 0
        end = np.argmax(y)
        start = len(probability_matrix) - 1 - start
        for _ in range(n):
            position = start
            while position != osk and position != end:
                position = np.random.choice(
                    len(probability_matrix[position]), p=probability_matrix[position])
            if position == end:
                successes += 1
        return successes

    def monte_carlo_simplified(self) -> np.float64:
        if not self.adjacency_matrix.any():
            raise ValueError("Probability matrix not set")
        if not self.y.any():
            raise ValueError("y not set")
        if not self.start_position:
            raise ValueError("Start position not set")

        adjacency_matrix: npfloatarray = self.adjacency_matrix
        y: npintarray = self.y
        start_position: int = self.start_position
        probability_matrix = self._build_probability_matrix(adjacency_matrix)

        trials: int = self.trials
        batches: int = self.batches
        batch_size: int = trials // batches
        osk: int = 0

        successes = 0
        with mp.Pool(mp.cpu_count()) as pool:
            results: list[AsyncResult] = [pool.apply_async(self._random_walk, args=(batch_size, probability_matrix, start_position, osk, y))
                                          for _ in range(batches)]
            pool.close()
            pool.join()

        successes = sum(result.get() for result in results)
        return np.float64(successes) / trials


def main():
    adjacency_matrix: npfloatarray = np.array([
        [1,     0,      0,      0,      0,      0],
        [-0.5,  1,      -0.5,   0,      0,      0],
        [0,     -0.5,   1,      -0.5,   0,      0],
        [0,     0,      -0.5,   1,      -0.5,   0],
        [0,     0,      0,      -0.5,   1,      -0.5],
        [0,     0,      0,      0,      0,      1],
    ])

    y: npintarray = np.array(
        [0, 0, 0, 0, 0, 1])  # p5, p4, p3, p2, p1, p0

    monte_carlo = MonteCarlo(trials=10000, batches=10, seed=123456789)
    monte_carlo\
        .set_adjacency_matrix(adjacency_matrix)\
        .set_y(y)\
        .set_start_position(3)

    result: np.float64 = monte_carlo.monte_carlo_simplified()
    print(result)


if __name__ == "__main__":
    main()
