import abc
import math
import tsplib95 as lib
import numpy as np
import matplotlib.pyplot as plt
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
        self.plot_route()
        return f"Path: {self.path}, Cost: {self.cost}"

    def plot_route(self):
        plt.figure(figsize=(12, 8))
        plt.scatter([i[0] for i in self.problem.coordinates[:]], [i[1] for i in self.problem.coordinates[:]],
                    c='black', zorder=1)

        for i, coord in enumerate(self.problem.coordinates):
            plt.annotate(str(i), xy=coord, xytext=(coord[0] + 10, coord[1]+10),
                         fontsize=18, ha='center', va='center')

        for j in range(-1, len(self.path) - 1, 1):
            print(len(self.path), len(self.problem.coordinates), j, self.path[j], self.problem.coordinates[j])
            start, end = self.path[j]-1, self.path[j + 1]-1
            plt.plot([self.problem.coordinates[start][0], self.problem.coordinates[end][0]],
                        [self.problem.coordinates[start][1], self.problem.coordinates[end][1]],
                        color='red', linewidth=2, zorder=2)

        plt.title(f"Total Route Distance: {self.cost:.2f}")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

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
        h_h_index = 0
        for n_h in nested_hulls:
            print(f"nested_hull #{h_h_index} : "
                  f"{[f'#{problem.coordinates.index(list(point)) + 1} -> [{point[0]}, {point[1]}]' for point in n_h]}")
            h_h_index += 1

        # Комбинируем вложенные выпуклые оболочки в один цикл
        combined_cycle = self.combine_nested_hulls_sequential(nested_hulls)
        
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

    def combine_nested_hulls_sequential(self, nested_hulls):
        # Начинаем с самой внешней оболочки
        if not nested_hulls:
            return []

        combined_hull = nested_hulls[0].tolist()

        # Последовательно объединяем каждую следующую оболочку
        for i in range(1, len(nested_hulls)):
            combined_hull = self.merge_two_hulls_combined(combined_hull, nested_hulls[i-1].tolist(), nested_hulls[i].tolist())

        return combined_hull

    def merge_two_hulls_combined(self, combined_hull, hull1, hull2):
        min_distance = float('inf')
        best_pair = None

        # Находим две ближайшие вершины между комбинированным циклом и новой оболочкой
        for point1 in hull1:
            for point2 in hull2:
                distance = self.euclidean_distance(point1, point2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = (point1, point2)

        point1, point2 = best_pair
        c_i = combined_hull.index(point1)
        k_j = hull2.index(point2)

        new_cycle = []

        # Добавляем в цикл все точки combined_hull до c_i
        new_cycle.extend(combined_hull[:c_i + 1])
        # Добавляем в цикл точку k_j и все точки hull2 после k_j, которых ещё нет в комбинированном цикле
        for idx in range(k_j, len(hull2)):
            if hull2[idx] not in new_cycle:
                new_cycle.append(hull2[idx])
        # Добавляем в цикл все точки hull2 до k_j, которых ещё нет в комбинированном цикле
        for idx in range(0, k_j):
            if hull2[idx] not in new_cycle:
                new_cycle.append(hull2[idx])
        # Добавляем оставшиеся точки combined_hull после c_i, которых ещё нет в новом цикле
        for idx in range(c_i + 1, len(combined_hull)):
            if combined_hull[idx] not in new_cycle:
                new_cycle.append(combined_hull[idx])

        return new_cycle

    @staticmethod
    def euclidean_distance(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
