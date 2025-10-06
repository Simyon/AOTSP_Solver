import argparse
from typing import Tuple
import numpy as np
from matplotlib.path import Path
import json
import csv
import os

# CONSTANT TSP
def generate_constant_tsp(n: int, constant_value: float = 10.0, seed: int = None) -> np.ndarray:
    """
    Генерирует матрицу расстояний для Constant TSP.

    Параметры:
        n (int): количество городов
        constant_value (float): расстояние между городами
        seed (int): сид генерации

    Возвращает:
        np.ndarray: Матрица расстояний (n x n)
    """
    if seed is not None:
        np.random.seed(seed)

    matrix = np.full((n, n), constant_value, dtype=float)

    # На диагонали нули
    np.fill_diagonal(matrix, 0)

    return matrix


# SMALL TSP
def generate_small_tsp(n: int, low: float = 0.1, high: float = 10, seed: int = None) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерация small TSP матрицы

    Параметры:
      n: число городов
      low, high: диапазон значений
      seed: сид генерации
    Возвращает:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
          C: Матрица расстояний (n x n)
          a: Вектор a (n,)
          b: Вектор b (n,)
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.random.uniform(low, high, size=2 * n)

    a = np.sort(values[:n])  # сортируем a
    b = values[n:]  # b не обязательно сортировать

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = min(a[i], b[j])

    # на диагоналях нули
    np.fill_diagonal(C, 0)

    return C, a, b


# CONVEX HULL TSP
def generate_convex_hull(n: int, radius: int = 10, noise: float = 0.0, seed=None):
    """
    Генерирует точки, лежащие на выпуклой оболочке.

    Параметры:
        n : int — количество точек
        radius   : float — радиус окружности, на которой лежат точки
        noise    : float — добавочный шум для радиуса, чтобы фигура была не идеальной окружностью
        seed     : int или None — случайное зерно

    Возвращает:
        List[List]
    """
    if seed is not None:
        np.random.seed(seed)

    # Равномерно распределяем углы
    angles = np.sort(np.random.rand(n) * 2 * np.pi)

    # Радиусы с небольшим шумом
    radii = radius + np.random.uniform(-noise, noise, n)

    # Преобразуем в координаты
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    points = np.column_stack((x, y))
    return points

def calculate_homothety_coefficient(n):
    """
    Вычисляет коэффициент гомотетии для правильного выпуклого многоугольника с n вершинами.
    """
    if n < 3:
        raise ValueError("n должно быть не менее 3")
    angle = 2 * np.pi / n
    sqrt_term = np.sqrt(2 * (1 - np.cos(angle)))
    coefficient = (2 + sqrt_term) / (2 - sqrt_term) - 1
    return coefficient

# MULTI-CONVEX HULL TSP
def generate_nested_convex_hulls(num_shells, n, base_radius=10, noise=0.0, seed=None):
    """
    Генерирует вложенные выпуклые оболочки.

    Параметры:
        num_shells: int — количество вложенных оболочек.
        n: int — количество точек на каждой оболочке.
        base_radius: float — радиус внешней оболочки.
        noise: float — добавочный шум для радиуса.
        seed: int или None — случайное зерно.

    Возвращает:
        List[np.ndarray] — список массивов точек для каждой оболочки.
    """
    if seed is not None:
        np.random.seed(seed)

    shells = []
    homothety_coeff = calculate_homothety_coefficient(n)
    constant = 13  

    for i in range(num_shells):
        current_radius = base_radius / (1 + homothety_coeff * constant * (i + 1))
        points = generate_convex_hull(n, current_radius, noise, seed)
        shells.append(points)

    return shells

# CONVEX HULL AND LINE TSP
def random_point_in_convex_polygon(vertices, seed=None):
    """
    Генерирует случайную точку внутри выпуклого многоугольника vertices (Nx2)
    с помощью случайных выпуклых коэффициентов.
    """
    if seed is not None:
        np.random.seed(seed)
    n = len(vertices)
    weights = np.random.rand(n)
    weights /= weights.sum()  # нормируем сумму весов до 1
    point = np.dot(weights, vertices)  # выпуклая комбинация вершин
    return point


def farthest_vertex_from_point(vertices, point):
    """
    Находит вершину многоугольника, максимально удалённую от заданной точки.
    """
    dists = np.linalg.norm(vertices - point, axis=1)
    idx = np.argmax(dists)
    return vertices[idx], idx


def random_point_near_vertex_within_polygon(vertices, vertex_idx, polygon_path, max_shift=1.0, seed=None):
    """
    Генерирует случайную точку рядом с вершиной vertex_idx, оставаясь внутри многоугольника.
    """
    if seed is not None:
        np.random.seed(seed)
    base = vertices[vertex_idx]
    for _ in range(100):  # попытки подобрать точку внутри
        shift = np.random.uniform(-max_shift, max_shift, 2)
        candidate = base + shift
        if polygon_path.contains_point(candidate):
            return candidate
    return base  # если не удалось, возвращаем вершину без смещения


def generate_line_points(vertices, n_points, max_fraction=1.0, noise=0.0, max_shift=1.0,
                         seed=None):
    """
    Генерирует линию внутри многоугольника между двумя достаточно удалёнными точками:
    - первая точка — случайная внутри многоугольника,
    - вторая — около максимально удалённой вершины от первой,
    - линия — часть отрезка между ними с длиной до max_fraction,
    - можно добавить шум по нормали.
    """
    if seed is not None:
        np.random.seed(seed)
    path = Path(vertices)

    # Первая точка — случайная внутри многоугольника
    p1 = random_point_in_convex_polygon(vertices, seed)

    # Вторая точка — около самой удалённой вершины от p1, внутри многоугольника
    far_vertex, far_idx = farthest_vertex_from_point(vertices, p1)
    p2 = random_point_near_vertex_within_polygon(vertices, far_idx, path, max_shift,
                                                 seed + 1 if seed is not None else None)

    # Ограничение длины линии случайным параметром t_max
    t_max = np.random.uniform(0.5, max_fraction)
    t = np.linspace(0, t_max, n_points)
    line_points = np.outer(1 - t, p1) + np.outer(t, p2)

    if noise > 0:
        edge_vec = p2 - p1
        normal_vec = np.array([-edge_vec[1], edge_vec[0]])
        normal_vec /= np.linalg.norm(normal_vec)
        noise_vals = np.random.uniform(-noise, noise, n_points)
        line_points += np.outer(noise_vals, normal_vec)

    return line_points


def generate_convex_hull_and_line_points(
        n_hull_points: int = 10,
        hull_radius: int | float = 10,
        noise: float = 0.3,
        n_line_points: int = 8,
        max_shift: float = 1.0,
        max_fraction: float = 1.0,
        seed: int | None = None
):
    """
    Основная функция генерации точек:
    - генерирует n_hull_points точек выпуклой оболочки,
    - генерирует линию с n_line_points точками внутри многоугольника с параметрами длины и шума,
    - возвращает объединённый массив точек, массив точек оболочки и массива точек линии.

    Параметры:
        n_hull_points : int
            Количество точек, формирующих выпуклую оболочку (многоугольник).

        hull_radius : int | float
            Радиус окружности, на которой располагаются точки выпуклой оболочки.

        noise : float
            Максимальная амплитуда случайного шума (смещения) точек от идеального положения.
            Применяется как к точкам оболочки, так и к точкам линии.

        n_line_points : int
            Количество точек, расположенных на линии внутри выпуклой оболочки.

        max_shift : float
            Максимальное поперечное смещение точек линии от прямой.

        max_fraction : float
            Максимальная относительная длина линии (в долях от максимального размера оболочки).

        seed : int | None
            Сид генерации
    """
    hull_points = generate_convex_hull(n_hull_points, hull_radius, noise, seed)
    center = np.mean(hull_points, axis=0)
    angles = np.arctan2(hull_points[:, 1] - center[1], hull_points[:, 0] - center[0])
    order = np.argsort(angles)
    hull_points_ordered = hull_points[order]

    line_points = generate_line_points(
        hull_points_ordered,
        n_line_points,
        max_fraction=max_fraction,
        noise=noise,
        max_shift=max_shift,
        seed=seed
    )

    points = np.vstack([hull_points_ordered, line_points])
    return points, hull_points_ordered, line_points


def split_points_safely(N, min_hull=3, min_line=1, hull_ratio_range=(0.3, 0.7), seed=None):
    """
    Разбивает N на n_hull и n_line с гарантией,
    что n_hull >= min_hull и n_line >= min_line.
    Если N слишком мал — корректирует пропорции.
    """
    if seed is not None:
        np.random.seed(seed)
    min_total = min_hull + min_line
    if N < min_total:
        raise ValueError(
            f"Общее количество точек N={N} слишком мало для минимальных требований: hull>= {min_hull}, line>= {min_line}")

    ratio = np.random.uniform(*hull_ratio_range)
    n_hull = int(N * ratio)
    n_line = N - n_hull

    if n_hull < min_hull:
        n_hull = min_hull
        n_line = N - n_hull
    if n_line < min_line:
        n_line = min_line
        n_hull = N - n_line

    if n_hull < min_hull or n_line < min_line:
        n_hull = max(min_hull, n_hull)
        n_line = max(min_line, n_line)
        total = n_hull + n_line
        if total > N:
            excess = total - N
            if n_hull > n_line:
                n_hull -= excess
            else:
                n_line -= excess
    return n_hull, n_line


# UPPER-TRIANGULAR TSP
def generate_upper_triangular_tsp(n: int, low: int = 1, high: int = 10, symmetric: bool = True, seed: int = None):
    """
    Генерация матрицы для Upper-Triangular TSP.

    n         : количество городов
    low, high : диапазон случайных расстояний (int)
    symmetric : симметричная ли матрица (евклидоподобный случай)
    seed      : сид генерации
    """
    if seed is not None:
        np.random.seed(seed)

    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            mat[i, j] = np.random.uniform(low, high)

    if symmetric:
        mat += mat.T

    return mat


def generate(
        n: int,
        case: str,
        seed: int = None,
        min_constant: float = None,
        max_constant: float = None,
        low_small: float = None,
        high_small: float = None,
        convex_radius: int = None,
        convex_noise: int = None,
        convex_shift: float = None,
        convex_fraction: float = None,
        convex_n_ratio: str = None,
        upper_low: int = None,
        upper_high: int = None,
        upper_symmetric: bool = None):
    result = None
    if case == "constant":
        min_constant = min_constant if min_constant else 0.1
        max_constant = max_constant if max_constant else 10.0
        constant_value = np.random.uniform(min_constant, max_constant)

        result = generate_constant_tsp(n, constant_value, seed)
        # dist_matrix = generate_constant_tsp(n, constant_value, seed)
        # return dist_matrix
    elif case == "small":
        low_small = low_small if low_small else 0.1
        high_small = high_small if high_small else 20.0

        result, a, b = generate_small_tsp(n, low=low_small, high=high_small, seed=seed)
        # matrix, a, b = generate_small_tsp(n, low=low_small, high=high_small, seed=seed)
        #
        # return matrix, a, b
    elif case == "convex":
        convex_radius = convex_radius if convex_radius else 10
        convex_noise = convex_noise if convex_noise else 0.0

        result = generate_convex_hull(n, convex_radius, convex_noise, seed=seed)
        # points = generate_convex_hull(n, convex_radius, convex_noise, seed=seed)
        #
        # return points
    elif case == "convex_and_line":
        if n <= 4:
            n_hull, n_line = n, 0
        else:
            convex_n_ratio = tuple(map(float, convex_n_ratio.split(","))) if convex_n_ratio else (0.4, 0.6)
            n_hull, n_line = split_points_safely(n, min_hull=3, min_line=2, hull_ratio_range=convex_n_ratio, seed=seed)

        convex_radius = convex_radius if convex_radius else 10
        convex_noise = convex_noise if convex_noise else 0.0
        convex_shift = convex_shift if convex_shift else 2.0
        convex_fraction = convex_fraction if convex_fraction else 1.0

        result, hull_pts, line_pts = generate_convex_hull_and_line_points(
            n_hull_points=n_hull,
            n_line_points=n_line,
            hull_radius=convex_radius,
            noise=convex_noise,
            max_shift=convex_shift,
            max_fraction=convex_fraction,
            seed=seed
        )

        # points, hull_pts, line_pts = generate_convex_hull_and_line_points(
        #     n_hull_points=n_hull,
        #     n_line_points=n_line,
        #     hull_radius=convex_radius,
        #     noise=convex_noise,
        #     max_shift=convex_shift,
        #     max_fraction=convex_fraction,
        #     seed=seed
        # )
        #
        # return points, hull_pts, line_pts
    elif case == "upper":
        upper_low = upper_low if upper_low else 1
        upper_high = upper_high if upper_high else 10
        upper_symmetric = upper_symmetric if upper_symmetric else False

        result = generate_upper_triangular_tsp(n, low=upper_low, high=upper_high, symmetric=upper_symmetric, seed=seed)
        # matrix = generate_upper_triangular_tsp(n, low=upper_low, high=upper_high, symmetric=upper_symmetric, seed=seed)
        #
        # return matrix

    return {
        "case": case,
        "n": n,
        "result": result.tolist() if isinstance(result, np.ndarray) else (
            [x.tolist() if isinstance(x, np.ndarray) else x for x in result]
            if isinstance(result, tuple) else result
        )
    }

def generate_multiple(n_examples, **kwargs):
    results = []
    for i in range(n_examples):
        if kwargs.get("seed") is not None:
            kwargs["seed"] = kwargs["seed"] + i  # меняем seed для уникальности
        res = generate(**kwargs)
        results.append(res)
    return results

def save_examples(examples, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".json":
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)

    elif ext == ".txt":
        with open(filename, "w", encoding="utf-8") as f:
            for i, ex in enumerate(examples):
                f.write(f"Example {i+1}:\n{ex}\n\n")

    elif ext == ".csv":
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["case", "n", "result"])
            for ex in examples:
                writer.writerow([ex["case"], ex["n"], json.dumps(ex["result"])])
    else:
        raise ValueError(f"Неподдерживаемое расширение файла: {ext}. Поддерживаются .json, .txt, .csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="Количество городов")
    parser.add_argument("--case", type=str, required=True,
                        choices=["constant", "small", "convex", "convex_and_line", "upper"],
                        help="Тип случая генерации TSP")
    parser.add_argument("--n_examples", type=int, default=1, help="Сколько примеров сгенерировать")
    parser.add_argument("--output", type=str, default=None, help="Файл для сохранения")
    parser.add_argument("--seed", type=int, default=None, help="Сид генератора случайных чисел")
    parser.add_argument("--min_constant", type=float, default=None, help="Минимальное значение (только для constant)")
    parser.add_argument("--max_constant", type=float, default=None, help="Максимальное значение (только для constant)")
    parser.add_argument("--low_small", type=float, default=None, help="Минимальное значение (только для small)")
    parser.add_argument("--high_small", type=float, default=None, help="Максимальное значение (только для small)")
    parser.add_argument("--convex_radius", type=int, default=None, help="Радиус (только для convex)")
    parser.add_argument("--convex_noise", type=float, default=None, help="Шум (только для convex)")
    parser.add_argument("--convex_shift", type=float, default=None, help="Смещение (только для convex_and_line)")
    parser.add_argument("--convex_fraction", type=float, default=None,
                        help="Максимальная относительная длина линии (только для convex_and_line)")
    parser.add_argument("--convex_n_ratio", type=str, default=None,
                        help="Доля городов на линии и на оболочке в формате кортежа x, y (только для convex_and_line)")
    parser.add_argument("--upper_low", type=int, default=None, help="Минимальное значение (только для upper)")
    parser.add_argument("--upper_high", type=int, default=None, help="Максимальное значение (только для upper)")
    parser.add_argument("--upper_symmetric", type=bool, default=None,
                        help="Возвращать симметричную матрицу или нет (только для upper)")

    args = parser.parse_args()

    kwargs = vars(args)
    n_examples = kwargs.pop("n_examples")
    output_file = kwargs.pop("output")

    examples = generate_multiple(n_examples, **kwargs)

    if output_file:
        save_examples(examples, output_file)
    else:
        print(examples)
    # out = main(args.n, args.case, args.seed, args.min_constant, args.max_constant,
    #            args.low_small, args.high_small, args.convex_radius, args.convex_noise,
    #            args.convex_shift, args.convex_fraction, args.convex_n_ratio)
    #
    # print(out)
