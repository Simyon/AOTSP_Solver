import abc 
import numpy as np

class TSP_Problem:
    def __init__(self, matrix):
        self.matrix = matrix

class ETSP_Problem(TSP_Problem):
    def __init__(self, matrix):
        super().__init__(matrix)
        self.vertices = self.matrix_to_vertices()

    def matrix_to_vertices(self):
        n = len(self.matrix)
        vertices = []
        for i in range(n):
            for j in range(i+1, n):
                if self.matrix[i][j] != 0:
                    x1, y1 = i, j
                    x2, y2 = j, i
                    vertices.append(((x1, y1), (x2, y2), self.matrix[i][j]))
        return vertices

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

    def merge_clusters(self, problems: List[TSP_Problem]) -> TSP_Problem:
        # TODO: Реализация метода merge_clusters
        pass

class ConvexHull_TSP_Solver(Cluster_TSP_Solver):
    def solve(self, problem: TSP_Problem) -> TSP_Solution:
        # TODO: Реализация метода solve с использованием выпуклой оболочки
        pass
