import abc
import math
import tsplib95 as lib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans, DBSCAN
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

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

    def point_on_sections(self, point_, secs_):
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        for sec in secs_:
            if abs(dist(point_, sec[0]) + dist(point_, sec[1]) - dist(sec[0], sec[1])) <= self.eps_on_sections:
                return True
        return False

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

    def extract_nested_hulls(self):
        remaining_points = np.array(self.coordinates)
        nested_hulls = []
        while len(remaining_points) > 3:
            hull = ConvexHull(remaining_points)
            hull_vertices = hull.vertices
            extended_hull_indices = self.points_on_hull(remaining_points, hull_vertices)
            extended_hull_points = remaining_points[extended_hull_indices]
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
        coords = problem.coordinates
        route, distance = self.solve_tsp_from_coords(coords)
        return TSP_Solution(problem, route, distance)

    def euclidean_distance_matrix(self, coords):
        n = len(coords)
        distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    distance = ((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)**0.5
                    distance_matrix[i][j] = distance
                    print(f"Distance between point {i} and point {j}: {distance}")
        return distance_matrix

    def solve_tsp_from_coords(self, coords, TIME_LIMIT=300):
        coords = [tuple(coord) for coord in coords]
        distance_matrix = self.euclidean_distance_matrix(coords)
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.time_limit.seconds = TIME_LIMIT
        search_parameters.log_search = True

        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            index = routing.Start(0)
            route_distance = 0
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
            route.append(manager.IndexToNode(index))
            return route, route_distance
        else:
            raise ValueError('Решение не найдено.')

class Cluster_TSP_Solver(TSP_Solver):
    def __init__(self, num_clusters: int = None):
        super().__init__()
        self.num_clusters = num_clusters
        self.clusters = {}

    def solve(self, problem: TSP_Problem) -> TSP_Solution:
        raise NotImplementedError("This method is not implemented yet.")

    def clusterize(self, coordinates, method='kmeans', eps=0.5, min_samples=5):
        if method == 'kmeans':
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            kmeans.fit(coordinates)
            self.clusters = {i: [] for i in range(self.num_clusters)}
            for idx, label in enumerate(kmeans.labels_):
                self.clusters[label].append(coordinates[idx])
        elif method == 'dbscan':
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(coordinates)
            labels = dbscan.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            self.clusters = {i: [] for i in range(n_clusters_)}
            for idx, label in enumerate(labels):
                if label != -1:
                    self.clusters[label].append(coordinates[idx])
        return self.clusters

    def plot_clusters(self):
        if not self.clusters:
            print("No clusters found.")
            return

        plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap('hsv', len(self.clusters))

        for i, cluster in self.clusters.items():
            cluster = np.array(cluster)
            if len(cluster) > 0:
                plt.scatter(cluster[:, 0], cluster[:, 1], color=colors(i), label=f'Cluster {i+1}')

        plt.title("Clusters of TSP Points")
        plt.xlabel('X')
        plt.ylabel('Y')
        if self.clusters:
            plt.legend()
        plt.grid(True)
        plt.show()

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
      nested_hulls = problem.extract_nested_hulls()
      combined_cycle = self.combine_nested_hulls_sequential(nested_hulls, problem.coordinates)

      path = []
      for point in combined_cycle:
          index = np.where((problem.coordinates == np.array(point)).all(axis=1))[0]
          if len(index) > 0:
              path.append(index[0] + 1)
          else:
              raise ValueError(f"Point {point} not found in coordinates")

      cost = 0
      for i in range(len(path)):
          from_city = path[i] - 1
          to_city = path[(i + 1) % len(path)] - 1
          cost += problem.matrix[from_city][to_city]

      return TSP_Solution(problem, path, cost)

class RouteMerger:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)
        self.distance_matrix = self.create_distance_matrix(self.coordinates)

    def create_distance_matrix(self, points):
        size = len(points)
        distance_matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(points[i] - points[j])
        return distance_matrix

    def merge_routes(self, route1, route2):
      min_distance = float('inf')
      best_i, best_j = -1, -1
      best_ij_flag = 0

      for i in range(len(route1) - 1):
          for j in range(len(route2) - 1):
              dist1 = (self.distance_matrix[route1[i]][route2[j]] +
                      self.distance_matrix[route2[j+1]][route1[i+1]] -
                      self.distance_matrix[route1[i]][route1[i+1]] -
                      self.distance_matrix[route2[j]][route2[j+1]])

              dist2 = (self.distance_matrix[route1[i]][route2[j+1]] +
                      self.distance_matrix[route2[j]][route1[i+1]] -
                      self.distance_matrix[route1[i]][route1[i+1]] -
                      self.distance_matrix[route2[j]][route2[j+1]])

              if min(dist1, dist2) < min_distance:
                  min_distance = min(dist1, dist2)
                  best_i, best_j = i, j
                  best_ij_flag = 1 if dist1 < dist2 else 0

      if best_ij_flag:
          new_route = route1[:best_i + 1] + list(reversed(route2[:best_j + 1])) + route2[best_j + 1:] + route1[best_i + 1:]
      else:
          new_route = route1[:best_i + 1] + route2[best_j + 1:] + route2[:best_j + 1] + route1[best_i + 1:]
          
      new_route = list(dict.fromkeys(new_route))
      return new_route

    def plot_all_routes(self, routes, total_distance, title):
      plt.figure(figsize=(12, 8))
      plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='black', zorder=1)

      colors = plt.cm.get_cmap('tab10', len(routes))
      for i, route in enumerate(routes):
          color = colors(i)
          for j in range(len(route) - 1):
              start, end = route[j], route[j+1]
              plt.plot([self.coordinates[start][0], self.coordinates[end][0]],
                      [self.coordinates[start][1], self.coordinates[end][1]],
                      color=color, linewidth=2, zorder=2)

      plt.title(f"{title}\nTotal Route Distance: {total_distance:.2f}")
      plt.xlabel('X')
      plt.ylabel('Y')
      plt.grid(True)
      plt.show()

    def merge_cluster_routes(self, cluster_routes, cluster_centroids):
      cluster_routes = [route.copy() for route in cluster_routes]
      cluster_centroids = [centroid.copy() for centroid in cluster_centroids]

      merge_step = 1
      while len(cluster_routes) > 1:
          min_distance = float('inf')
          best_pair = None

          for i in range(len(cluster_centroids) - 1):
              for j in range(i + 1, len(cluster_centroids)):
                  dist = np.linalg.norm(cluster_centroids[i] - cluster_centroids[j])
                  if dist < min_distance:
                      min_distance = dist
                      best_pair = (i, j)

          if best_pair:
              i, j = best_pair
              route1 = cluster_routes.pop(i)
              route2 = cluster_routes.pop(j - 1 if j > i else j)
              merged_route = self.merge_routes(route1, route2)

              all_points = set(route1 + route2)
              if len(merged_route) != len(all_points):
                  missing_points = list(all_points - set(merged_route))
                  merged_route.extend(missing_points)

              cluster_routes.append(merged_route)

              new_centroid = np.mean([self.coordinates[point] for point in merged_route], axis=0)
              cluster_centroids.pop(i)
              cluster_centroids.pop(j - 1 if j > i else j)
              cluster_centroids.append(new_centroid)

              total_distance = sum(self.calculate_route_distance(route) for route in cluster_routes)
              self.plot_all_routes(cluster_routes, total_distance, f'Merged Routes After Step {merge_step}')
              merge_step += 1

      final_route = cluster_routes[0]
      return final_route

    def calculate_route_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i+1]]
        total_distance += self.distance_matrix[route[-1]][route[0]]
        return total_distance

class GridTSPSolver:
  def __init__(self, coordinates, num_squares=4):
        self.coordinates = np.array(coordinates)  
        self.num_squares = num_squares
        self.squares = self.divide_into_squares(self.coordinates, num_squares)

  def divide_into_squares(self, coordinates, num_squares):
      min_x, min_y = np.min(coordinates, axis=0)
      max_x, max_y = np.max(coordinates, axis=0)

      x_step = (max_x - min_x) / num_squares
      y_step = (max_y - min_y) / num_squares

      squares = []
      for i in range(num_squares):
          for j in range(num_squares):
              x_min, x_max = min_x + i * x_step, min_x + (i + 1) * x_step
              y_min, y_max = min_y + j * y_step, min_y + (j + 1) * y_step

              square_coords = [coord for coord in coordinates
                                if x_min <= coord[0] < x_max and y_min <= coord[1] < y_max]
              squares.append(square_coords)

      return squares

  def solve_tsp_in_squares(self):
      convex_hull_solver = ConvexHull_TSP_Solver(num_clusters=1)
      routes = []

      for square in self.squares:
          if len(square) > 0:
              problem = ETSP_Problem.from_coordinates(square)
              solution = convex_hull_solver.solve(problem)
              routes.append(solution.path)

      return routes

  def merge_routes(self, routes):
      route_merger = RouteMerger(self.coordinates)
      final_route = routes[0]

      for route in routes[1:]:
          final_route = route_merger.merge_routes(final_route, route)

      return final_route

  def plot_routes(self, routes, title):
    plt.figure(figsize=(12, 8))
    plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='black', zorder=1)

    colors = plt.cm.get_cmap('tab10', len(routes))
    for i, route in enumerate(routes):
        color = colors(i)
        for j in range(len(route) - 1):
            start, end = route[j], route[j+1]
            plt.plot([self.coordinates[start][0], self.coordinates[end][0]],
                     [self.coordinates[start][1], self.coordinates[end][1]],
                     color=color, linewidth=2, zorder=2)

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()

  def solve_and_plot(self):
      routes = self.solve_tsp_in_squares()
      final_route = self.merge_routes(routes)
      total_distance = self.calculate_route_distance(final_route)
      self.plot_routes([final_route], f'Final Merged Route\nTotal Route Distance: {total_distance:.2f}')

  def calculate_route_distance(self, route):
    coordinates = np.array(self.coordinates)  
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += np.linalg.norm(coordinates[route[i]] - coordinates[route[i+1]])
    total_distance += np.linalg.norm(coordinates[route[-1]] - coordinates[route[0]])
    return total_distance
