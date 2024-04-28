from multiprocessing.pool import AsyncResult
import numpy as np
from typing import Any, TypeAlias
import multiprocessing as mp

npintarray: TypeAlias = np.ndarray[np.int32, np.dtype[Any]]
npfloatarray: TypeAlias = np.ndarray[np.float64, np.dtype[Any]]


class MonteCarlo:
    '''
    A class simulating a random walk in a graph.\n
    The graph is represented by an adjacency matrix
    '''

    def __init__(
            self,
            trials: int,
            batches: int,
            seed: int,
            adjacency_matrix: npfloatarray,
            y: npintarray,
            starting_position: int,
            OSKs: npintarray
    ) -> None:
        '''
        Parameters:
        -------
        trials: number of trials
        batches: number of batches for multiprocessing
        seed: seed for random number generator
        adjacency_matrix: adjacency matrix of the graph
        y: list of correct end nodes
        starting_position: starting position
        OSKs: list of wrong end nodes
        '''

        self.trials: int = trials
        self.batches: int = batches
        self.seed: int = seed
        self.adjacency_matrix: npfloatarray = adjacency_matrix
        self.y: npintarray = y
        self.starting_position = starting_position
        self.OSKs = OSKs

    def build_probability_matrix(self, adjacency_matrix: npfloatarray) -> Any:
        '''
        build_probability_matrix(adjacency_matrix: ndarray[np.float64, np.dtype[Any]]) -> ndarray[np.float64, np.dtype[Any]]\n
        Builds a probability matrix from adjacency matrix.

        Probability matrix is a matrix where each row is a probability distribution.\n
        It is built by subtracting the identity matrix from the adjacency matrix
        negating the result, and adding back 1 to the diagonal if the row is all zeros (exit node).

        Example:
        ----------------
        adjacency_matrix = np.array([
            [1,     0,      0,      0,      0,      0],\n
            [-0.5,  1,      -0.5,   0,      0,      0],\n
            [0,     -0.5,   1,      -0.5,   0,      0],\n
            [0,     0,      -0.5,   1,      -0.5,   0],\n
            [0,     0,      0,      -0.5,   1,      -0.5],\n
            [0,     0,      0,      0,      0,      1],\n
        ])

        build_probability_matrix(adjacency_matrix) = np.array([
            [1,     0,      0,      0,      0,      0],\n
            [0.5,   0,      0.5,    0,      0,      0],\n
            [0,     0.5,    0,      0.5,    0,      0],\n
            [0,     0,      0.5,    0,      0.5,    0],\n
            [0,     0,      0,      0.5,    0,      0.5],\n
            [0,     0,      0,      0,      0,      1],\n
        ])

        Returns
        -------
        probability_matrix: probability matrix
        '''
        probability_matrix = adjacency_matrix - \
            np.identity(adjacency_matrix.shape[0], dtype=np.float64)
        probability_matrix: npfloatarray = np.negative(probability_matrix)
        for i in range(probability_matrix.shape[0]):
            for j in range(probability_matrix.shape[1]):
                if np.all(probability_matrix[i] == 0):
                    probability_matrix[i][i] = 1
        return probability_matrix

    def random_walk_single_start(self, n: int, probability_matrix: npfloatarray) -> int:
        '''
        random_walk_single_start(n: int, probability_matrix: np.ndarray[np.float64, np.dtype[Any]]) -> int

        Simulation of a random walk with a single starting position.

        Parameters:
        -------
        n: number of trials
        probability_matrix: probability matrix

        Returns:
        -------
        successes: number of successes (reaching a correct end node)
        '''
        np.random.seed(123456789)
        successes = 0
        ends: npintarray = np.where(self.y == 1)[0]
        start: int = self.starting_position
        for _ in range(n):
            position = start
            while position not in self.OSKs and position not in ends:
                position = np.random.choice(
                    len(probability_matrix[position]), p=probability_matrix[position])
            if position in ends:
                successes += 1
        return successes

    def monte_carlo_no_multiprocessing(self) -> np.float64:
        '''
        monte_carlo_no_multiprocessing() -> np.float64

        Runs the Monte Carlo simulation without multiprocessing

        Returns:
        -------
        result: probability of reaching a correct end node
        '''
        adjacency_matrix: npfloatarray = self.adjacency_matrix
        probability_matrix = self.build_probability_matrix(adjacency_matrix)
        trials: int = self.trials
        successes: int = self.random_walk_single_start(
            trials, probability_matrix)
        return np.float64(successes) / trials

    def monte_carlo(self) -> np.float64:
        '''
        monte_carlo() -> np.float64

        Runs the Monte Carlo simulation with multiprocessing

        Returns:
        -------
        result: probability of reaching a correct end node

        Throws:
        -------
        ValueError: if not in debug mode
        '''

        adjacency_matrix: npfloatarray = self.adjacency_matrix
        probability_matrix = self.build_probability_matrix(adjacency_matrix)

        trials: int = self.trials
        batches: int = self.batches
        batch_size: int = trials // batches

        successes = 0
        with mp.Pool(mp.cpu_count()) as pool:
            results: list[AsyncResult] = [pool.apply_async(self.random_walk_single_start, args=(batch_size, probability_matrix))
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

    y: npintarray = np.array([1, 0, 0, 0, 0, 0])   # p0, p1, p2, p3, p4, p5
    trials: int = 1000000
    batches: int = 10
    seed: int = 123456789
    starting_position: int = 3
    OSKs: npintarray = np.array([5])

    monte_carlo = MonteCarlo(
        trials=trials,
        batches=batches,
        seed=seed,
        adjacency_matrix=adjacency_matrix,
        y=y,
        starting_position=starting_position,
        OSKs=OSKs
    )
    print(
        f'Running Monte Carlo simulation with {trials=}, {batches=}, {seed=}')
    print(f'Starting position: {starting_position}, OSKs: {OSKs}, y: {y}')
    result: np.float64 = monte_carlo.monte_carlo()
    print(f'Probability of reaching a correct end node: {result:.4f}')


if __name__ == "__main__":
    main()
