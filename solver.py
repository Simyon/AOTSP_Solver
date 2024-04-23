import abc
import math
import tsplib95 as lib
import numpy as np

class TSP_Problem:
    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def load_tsplib(cls, path: str):
        matrix = []
        # TODO: Реализация чтения матрицы из tsplib
        return cls(matrix)

class ETSP_Problem(TSP_Problem):
    def __init__(self, matrix, coordinates = None):
        super().__init__(matrix)
        if not coordinates:
            self.coordinates = self.matrix_to_coordinates()
        else:
            self.coordinates = coordinates

    def matrix_to_coordinates(self):
        n = len(self.matrix)
        coordinates = []
        for i in range(n):
            for j in range(i+1, n):
                if self.matrix[i][j] != 0:
                    x1, y1 = i, j
                    x2, y2 = j, i
                    coordinates.append(((x1, y1), (x2, y2), self.matrix[i][j]))
        return coordinates

    @classmethod
    def load_tsplib(cls, path: str):
        problem = lib.load(path)
        coordinates = [problem.node_coords[i] for i in list(problem.get_nodes())]
        n = len(coordinates)
        matrix = np.zeros((n, n))
        for from_counter, from_node in enumerate(coordinates):
            for to_counter, to_node in enumerate(coordinates):
                if from_counter == to_counter:
                    matrix[from_counter, to_counter] = 0
                else:
                    # Euclidean distance
                    matrix[from_counter, to_counter] = int(math.hypot((from_node[0] * 1000 - to_node[0] * 1000),
                                                                      (from_node[1] * 1000 - to_node[1] * 1000)))
        return cls(matrix, coordinates)

class TSP_Solution:
    def __init__(self, problem, path, cost):
        self.problem = problem # object of TSP_Problem
        self.path = path
        self.cost = cost

    def __str__(self):
        return f"Path: {self.path}, Cost: {self.cost}"

class TSP_Solver(abc.ABC):
    def __init__(self):
        self.time = 0.0

    @abc.abstractmethod
    def solve(self, problem: TSP_Problem) -> TSP_Solution:
        pass

class OR_Tools_Solver(TSP_Solver):
    def solve(self, problem: TSP_Problem) -> TSP_Solution:
        # TODO: Реализация метода solve с использованием OR-Tools
        pass

class Cluster_TSP_Solver(TSP_Solver):
    def __init__(self, num_clusters: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.clusters = {}

    def merge_clusters(self, problems: list[TSP_Problem]) -> TSP_Problem:
        # TODO: Реализация метода merge_clusters
        pass

class ConvexHull_TSP_Solver(Cluster_TSP_Solver):
    def solve(self, problem: TSP_Problem) -> TSP_Solution:
        # TODO: Реализация метода solve с использованием выпуклой оболочки
        pass
