import sys
import importlib
import numpy as np
from importlib import reload
from sklearn.preprocessing import StandardScaler

sys.path.append('/content/sample_data')
 
import solver
reload(solver) # Обновление для Collab
from solver import ETSP_Problem, Cluster_TSP_Solver, OR_Tools_Solver, RouteMerger, GridTSPSolver


problem = ETSP_Problem.load_tsplib('datasets/berlin52.tsp')
num_clusters = 9
solver_instance = Cluster_TSP_Solver(num_clusters)
coordinates = problem.coordinates

# Кластеризация с использованием k-means
solver_instance.clusterize(coordinates, method='kmeans')
solver_instance.plot_clusters()

'''
# Нормализация координат
scaler = StandardScaler()
normalized_coordinates = scaler.fit_transform(coordinates)

# Кластеризация с использованием DBSCAN
solver_instance.clusterize(normalized_coordinates, method='dbscan', eps=0.4, min_samples=2)
solver_instance.plot_clusters()
'''

or_tools_solver = OR_Tools_Solver()
cluster_routes = []
cluster_centroids = []
initial_distances = []

# Подсчёт длительности для каждого цикла
def calculate_route_distance(coords, route):
    distance = 0.0
    num_points = len(route)
    for i in range(num_points):
        from_point = route[i]
        to_point = route[(i + 1) % num_points]  
        distance += np.linalg.norm(np.array(coords[from_point]) - np.array(coords[to_point]))
    return distance
    
for cluster_id, cluster_coords in solver_instance.clusters.items():
    print(f"Решение задачи коммивояжера для кластера {cluster_id + 1}")
    try:
        route, distance = or_tools_solver.solve_tsp_from_coords(cluster_coords)
        print(f"Маршрут: {route}, Расстояние: {distance}")        
        manual_distance = calculate_route_distance(cluster_coords, route)
        print(f"Ручное расстояние: {manual_distance}")
        cluster_routes.append(route)
        centroid = np.mean(cluster_coords, axis=0)
        cluster_centroids.append(centroid)
        initial_distances.append(manual_distance)  
    except Exception as e:
        print(f"Ошибка при решении задачи коммивояжера для кластера {cluster_id + 1}: {e}")
print("Initial Distances:", initial_distances)

# Визуализация суммы всех расстояний до сшивки
plt.figure(figsize=(12, 6))
plt.bar(range(len(initial_distances)), initial_distances, color='skyblue')
plt.title('Total Distances Before Merging')
plt.xlabel('Cluster ID')
plt.ylabel('Distance')
plt.grid(True)
plt.show()


route_merger = RouteMerger(coordinates)

# Сшивка маршрутов
final_route = route_merger.merge_cluster_routes(cluster_routes, cluster_centroids)

# Визуализация финального маршрута после сшивки
total_distance = route_merger.calculate_route_distance(final_route)
route_merger.plot_all_routes([final_route], total_distance, 'Final Merged Route')


grid_solver = GridTSPSolver(coordinates, num_squares=4)
print("===========================================================\nGRID\n===========================================================")
grid_routes = grid_solver.solve_tsp_in_squares()
grid_final_route = grid_solver.merge_routes(grid_routes)
grid_total_distance = grid_solver.calculate_route_distance(grid_final_route)

# Визуализация маршрута после оптимизации
grid_solver.plot_routes([grid_final_route], 'Optimized Route After Grid Solver')

# Сравнение расстояний
distances = {
    'Before Merging': sum(initial_distances),
    'After Merging': total_distance,
    'After Grid Optimization': grid_total_distance
}
plt.figure(figsize=(12, 6))
plt.bar(distances.keys(), distances.values(), color=['skyblue', 'orange', 'green'])
plt.title('Comparison of Total Distances')
plt.xlabel('Stage')
plt.ylabel('Total Distance')
plt.grid(True)
plt.show()
