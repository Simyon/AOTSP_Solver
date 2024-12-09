import os
from solver import ETSP_Problem, ConvexHull_TSP_Solver

def test_convex_hull_solver():
    data_dir = os.path.join(os.getcwd(), 'data')
    tsp_file_path = os.path.join(data_dir, 'berlin52.tsp')

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

if __name__ == "__main__":
    print("Тест решателя ConvexHull_TSP_Solver:")
    test_convex_hull_solver()
