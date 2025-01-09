import os
from solver import ETSP_Problem, ConvexHull_TSP_Solver

'''
для google colab
import importlib
from sample_data import solver
importlib.reload(solver)
from sample_data.solver import ETSP_Problem, ConvexHull_TSP_Solver
'''

def plot_nested_hulls(nested_hulls):
    plt.figure(figsize=(10, 8))
    
    # Список цветов для разных оболочек
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, hull in enumerate(nested_hulls):
        hull_points = np.array(hull)
        x, y = hull_points[:, 0], hull_points[:, 1]

        # Замыкаем контур для отображения
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        plt.plot(x, y, label=f'Hull {i + 1}', color=colors[i % len(colors)], linewidth=2)
        plt.scatter(hull_points[:, 0], hull_points[:, 1], color=colors[i % len(colors)], zorder=3)

    plt.title('Nested Convex Hulls')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_convex_hull_solver():
    tsp_file_path = 'berlin52.tsp'

    if not os.path.exists(tsp_file_path):
        print(f"Error: Файл {tsp_file_path} не найден.")
        return
    problem = ETSP_Problem.load_tsplib(tsp_file_path)

    # Создаем решатель ConvexHull_TSP_Solver и решаем задачу
    solver = ConvexHull_TSP_Solver(num_clusters=1)  # num_clusters в данном случае не используется
    solution = solver.solve(problem)
    print("Решение задачи TSP:")
    print(solution)
    solution.plot_route()

def test_convex_hull_solver_from_coordinates():
    coordinates = [
    [8, 89], [8, 81], [8, 73], [8, 65], [8, 57], [8, 49], [8, 41], [16, 17], [44, 11], [56, 9], [72, 9], [80, 9], [92, 9], [120, 9], [128, 9], [136, 9], [148, 9], [162, 9], [252, 21], [260, 29], [284, 53], [284, 61], [284, 69], [288, 109], [288, 129], [288, 149], [236, 169], [228, 169], [220, 169], [212, 169], [204, 169], [196, 169], [188, 169], [172, 169], [164, 169], [156, 169], [148, 169], [140, 169], [132, 169], [124, 169], [104, 169], [56, 169], [40, 169], [32, 169], [8, 109], [8, 97],
    [16, 57], [16, 25], [24, 17], [32, 17], [56, 17], [104, 17], [172, 21], [180, 21], [228, 21], [236, 21], [260, 37], [276, 53], [284, 77], [284, 85], [284, 93], [284, 101], [280, 133], [256, 157], [228, 161], [196, 161], [90, 165], [64, 165], [40, 161], [32, 161], [16, 109], [16, 97],
    [24, 45], [24, 25], [32, 25], [64, 21], [124, 21], [132, 21], [228, 29], [236, 29], [260, 45], [276, 61], [276, 69], [276, 77], [280, 109], [270, 133], [246, 157], [116, 161], [104, 161], [56, 161], [32, 153], [32, 145], [24, 89]
]
    problem = ETSP_Problem.from_coordinates(coordinates)

    # Создаем решатель ConvexHull_TSP_Solver и решаем задачу
    solver = ConvexHull_TSP_Solver(num_clusters=1)  # num_clusters в данном случае не используется
    solution = solver.solve(problem)
    print("Решение задачи TSP:")
    print(solution)
    solution.plot_route()
    nested_hulls = problem.extract_nested_hulls()
    plot_nested_hulls(nested_hulls)

if __name__ == "__main__":
    print("Тест решателя ConvexHull_TSP_Solver:")
    test_convex_hull_solver()
