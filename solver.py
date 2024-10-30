import abc
import math
import tsplib95 as lib
import numpy as np
from scipy.spatial import ConvexHull   

class TSP_Problem:
    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def load_tsplib(cls, path: str):
        matrix = []
        return cls(matrix)

class ETSP_Problem(TSP_Problem):
    def __init__(self, matrix, coordinates=None):
        super().__init__(matrix)
        if coordinates is None:
            self.coordinates = self.matrix_to_coordinates()
        else:
            self.coordinates = coordinates

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
                    matrix[from_counter, to_counter] = cls.euclidean_distance(from_node, to_node)

        return cls(matrix, coordinates)

    @staticmethod
    def euclidean_distance(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def extract_convex_hull(self):
        points = np.array(self.coordinates)
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return hull_points, hull.vertices

    def extract_nested_hulls(self):
        remaining_points = np.array(self.coordinates)
        nested_hulls = []

        while len(remaining_points) > 3:
            hull = ConvexHull(remaining_points)
            hull_points = remaining_points[hull.vertices]
            nested_hulls.append(hull_points)
            remaining_points = np.delete(remaining_points, hull.vertices, axis=0)

        if len(remaining_points) > 0:
            nested_hulls.append(remaining_points)

        return nested_hulls

    def combine_nested_hulls(self, nested_hulls):
        cycle = []

        # Начинаем с внешней оболочки
        for hull in nested_hulls:
            for point in hull:
                if list(point) not in cycle:
                    cycle.append(list(point))

        return cycle

class TSP_Solution:
    def __init__(self, problem, path, cost):
        self.problem = problem
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
    def __init__(self, num_clusters: int):
        super().__init__(num_clusters)

    def solve(self, problem: ETSP_Problem) -> TSP_Solution:
        # Извлекаем вложенные выпуклые оболочки
        nested_hulls = problem.extract_nested_hulls()

        # Комбинируем вложенные выпуклые оболочки в один цикл
        combined_cycle = problem.combine_nested_hulls(nested_hulls)
        
        path = []
        for point in combined_cycle:
            index = problem.coordinates.index(list(point))
            path.append(index + 1)  

        # Вычисляем стоимость маршрута
        cost = 0
        for i in range(len(path)):
            from_city = path[i] - 1
            to_city = path[(i + 1) % len(path)] - 1
            cost += problem.matrix[from_city][to_city]

        return TSP_Solution(problem, path, cost)
