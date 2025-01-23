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
    def __init__(self, matrix, coordinates=None, eps_on_sections=0.1):
        super().__init__(matrix)
        self.eps_on_sections = eps_on_sections
        if coordinates is None:
            self.coordinates = self.matrix_to_coordinates()
        else:
            self.coordinates = coordinates

    @classmethod
    def load_tsplib(cls, path: str, eps_on_sections=0.1):
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

        return cls(matrix, coordinates, eps_on_sections)

    @classmethod
    def from_coordinates(cls, coordinates, eps_on_sections=0.1):
        n = len(coordinates)
        matrix = np.zeros((n, n))

        for from_counter, from_node in enumerate(coordinates):
            for to_counter, to_node in enumerate(coordinates):
                if from_counter == to_counter:
                    matrix[from_counter, to_counter] = 0
                else:
                    matrix[from_counter, to_counter] = cls.euclidean_distance(from_node, to_node)

        return cls(matrix, coordinates, eps_on_sections)

    @staticmethod
    def euclidean_distance(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    # Функция для проверки, находится ли точка на отрезках выпуклой оболочки
    def point_on_sections(self, point_, secs_):
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        for sec in secs_:
            if abs(dist(point_, sec[0]) + dist(point_, sec[1]) - dist(sec[0], sec[1])) <= self.eps_on_sections:
                return True

        return False

    # Функция для определения всех точек, лежащих на выпуклой оболочке с учетом точек на гранях
    def points_on_hull(self, points_, hull_):
        hull_points = set(hull_)
        secs = []
        for i in range(len(hull_)):
            secs.append((points_[hull_[i]], points_[hull_[i-1]]))

        for num, point in enumerate(points_):
            if num not in hull_points and self.point_on_sections(point, secs):
                hull_points.add(num)

        return np.array(list(hull_points))

    def sort_points_by_polar_angle(self, points):
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])

        def polar_angle(point):
            return math.atan2(point[1] - center_y, point[0] - center_x)

        sorted_points = sorted(points, key=polar_angle)
        return np.array(sorted_points)

    # Основная функция для выделения вложенных выпуклых оболочек
    def extract_nested_hulls(self):
        remaining_points = np.array(self.coordinates)
        nested_hulls = []

        while len(remaining_points) > 3:
            hull = ConvexHull(remaining_points)
            hull_vertices = hull.vertices

            # Получаем все индексы точек на границе, включая лежащие на отрезках
            extended_hull_indices = self.points_on_hull(remaining_points, hull_vertices)
            extended_hull_points = remaining_points[extended_hull_indices]

            # Сортируем точки оболочки по полярному углу
            sorted_hull_points = self.sort_points_by_polar_angle(extended_hull_points)

            nested_hulls.append(sorted_hull_points)
            remaining_points = np.delete(remaining_points, extended_hull_indices, axis=0)

        if len(remaining_points) > 0:
            nested_hulls.append(remaining_points)

        return nested_hulls

class TSP_Solution:
    def __init__(self, problem, path, cost):
        self.problem = problem
        self.path = path
        self.cost = cost

    def __str__(self):
        return f"Path: {self.path}, Cost: {self.cost}"

    def plot_route(self):
        plt.figure(figsize=(12, 8))
        plt.scatter([i[0] for i in self.problem.coordinates[:]], [i[1] for i in self.problem.coordinates[:]],
                    c='black', zorder=1)

        for i, coord in enumerate(self.problem.coordinates):
            plt.annotate(str(i+1), xy=coord, xytext=(coord[0] + 10, coord[1]+10),
                         fontsize=18, ha='center', va='center')

        for j in range(-1, len(self.path) - 1, 1):
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

    def create_distance_matrix(self, points):
        points = np.array(points)
        size = len(points)
        distance_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(points[i] - points[j])
        return distance_matrix

    # Функция для объединения двух кластеров
    def merge_routes(self, route1, route2, distance_matrix):
        coordinates_to_index = {tuple(coord): idx for idx, coord in enumerate(self.coordinates)}

        min_distance = float('inf')
        best_i, best_j = -1, -1
        ij_flag = 0

        for i in range(len(route1) - 1):
            for j in range(len(route2) - 1):
                index1_i = coordinates_to_index[tuple(route1[i])]
                index1_i1 = coordinates_to_index[tuple(route1[i + 1])]
                index2_j = coordinates_to_index[tuple(route2[j])]
                index2_j1 = coordinates_to_index[tuple(route2[j + 1])]

                dist1 = distance_matrix[index1_i][index2_j] + distance_matrix[index2_j1][index1_i1] - distance_matrix[index1_i][index1_i1] - distance_matrix[index2_j][index2_j1]
                dist2 = distance_matrix[index1_i][index2_j1] + distance_matrix[index2_j][index1_i1] - distance_matrix[index1_i][index1_i1] - distance_matrix[index2_j][index2_j1]

                if min(dist1, dist2) < min_distance:
                    min_distance = min(dist1, dist2)
                    best_i, best_j = i, j
                    ij_flag = 1 if dist1 < dist2 else 0

        if ij_flag:
            new_route = route1[:best_i + 1] + list(reversed(route2[:best_j + 1])) + list(reversed(route2[best_j + 1:])) + route1[best_i + 1:]
        else:
            new_route = route1[:best_i + 1] + route2[best_j + 1:] + route2[:best_j + 1] + route1[best_i + 1:]

        new_route = list(dict.fromkeys(tuple(point) for point in new_route))
        return new_route

    def combine_nested_hulls_sequential(self, nested_hulls, coordinates):
        self.coordinates = coordinates
        if not nested_hulls:
            return []

        combined_hull = nested_hulls[0].tolist()
        distance_matrix = self.create_distance_matrix(coordinates)

        for i in range(1, len(nested_hulls)):
            combined_hull = self.merge_routes(combined_hull, nested_hulls[i].tolist(), distance_matrix)

        return combined_hull

    def solve(self, problem: ETSP_Problem) -> TSP_Solution:
        # Извлекаем вложенные выпуклые оболочки
        nested_hulls = problem.extract_nested_hulls()
        h_h_index = 0
        for n_h in nested_hulls:
            print(f"nested_hull #{h_h_index} : "
                  f"{[f'#{problem.coordinates.index(list(point)) + 1} -> [{point[0]}, {point[1]}]' for point in n_h]}")
            h_h_index += 1

        # Комбинируем вложенные выпуклые оболочки в один цикл
        combined_cycle = self.combine_nested_hulls_sequential(nested_hulls, problem.coordinates)

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

    @staticmethod
    def euclidean_distance(coord1, coord2):
        x1, y1 = coord1
        x2, y2 = coord2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
