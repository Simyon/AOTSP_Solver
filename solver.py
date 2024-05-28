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
    def __init__(self, num_clusters: int):
        super().__init__()
        self.num_clusters = num_clusters
        self.clusters = {}

    def nearest_neighbour_dot(self, point, convex_hull, forbidden_vertices = []):
        index_min_vertex = 0
        for i in range(len(convex_hull)):
            if i not in forbidden_vertices and self.euclid_dist(point, convex_hull[i]) < self.euclid_dist(point, convex_hull[index_min_vertex]):
                index_min_vertex = i
        return index_min_vertex

    def find_closest_vertices(self, convex_hull_1, convex_hull_2, forbidden_vertices = []):
        if len(convex_hull_2) == 1: self.nearest_neighbour_dot(convex_hull_2[0], convex_hull_1)

        min_i, min_ii, min_j, min_jj = 0, 1, 0, 1
        min_distance = self.euclid_dist(convex_hull_1[min_i], convex_hull_2[min_j]) + self.euclid_dist(convex_hull_1[min_ii], convex_hull_1[min_jj])
        for i in range(len(convex_hull_1)):
            c_i, c_ii = convex_hull_1[i], convex_hull_1[(i+1) % len(convex_hull_1)]
            if c_i in forbidden_vertices or c_ii in forbidden_vertices: continue
            for j in range(len(convex_hull_2)):
                c_j, c_jj = convex_hull_2[j], convex_hull_2[(j+1) % len(convex_hull_2)]
                c_distance = self.euclid_dist(c_i, c_j) + self.euclid_dist(c_ii, c_jj)
                if min_distance > c_distance:
                    min_i, min_ii, min_j, min_jj = i, (i+1) % len(convex_hull_1), j, (j+1) % len(convex_hull_2)
                    min_distance = c_distance

        return min_distance, [min_i, min_ii], [min_j, min_jj]

    def solve_tsp(self, convex_hullz):
        ch_costs = []
        patch_costs = []
        forbidden_vertices = {}  # changed to dict, no need for deep copy

        for i in range(len(convex_hullz) - 1):
            distance, near_1, near_2 = self.find_closest_vertices(convex_hullz[i], convex_hullz[i + 1], forbidden_vertices.keys())
            patch_costs.append(distance)
            forbidden_vertices[i] = near_1  # changed to tuple assignment
            forbidden_vertices[i + 1] = near_2

        for i, c_h in enumerate(convex_hullz):  # changed to enumerate
            pts = ConvexHull(c_h)
            ex_edge = 0
            if i in forbidden_vertices:
                f_A, f_B = c_h[forbidden_vertices[i][0]], c_h[forbidden_vertices[i][1]]  # changed to tuple unpacking
                ex_edge = self.euclid_dist(f_A, f_B)
            local_cost = self.hull_distance(pts) - ex_edge
            ch_costs.append(local_cost)

        cost = sum(ch_costs) + sum(patch_costs)
        print("ch_costs", ch_costs)
        print("patch_costs", patch_costs)
        print("cost", cost)
        return cost, ch_costs, patch_costs  # added return statement

    def solve(self, problem: TSP_Problem) -> TSP_Solution:
        cost, ch_costs, patch_costs = self.solve_tsp(convex_hullz)  # use solve_tsp to get cost
        path = []  # TODO: generate path from convex_hullz
        return TSP_Solution(problem, path, cost)
