import numpy as np

def estimate_foe(flow_vectors, inlier_threshold=3, num_iterations=100):
    """
    Оцінює точку сходження (FOE) за допомогою методу RANSAC для векторів потоку.
    
    Параметри:
    flow_vectors (list or ndarray): Список/масив векторів потоку (x, y).
    inlier_threshold (float): Поріг для визначення внутрішніх точок.
    num_iterations (int): Кількість ітерацій для виконання RANSAC.

    Повертає:
    best_foe (tuple): Краща оцінка точки сходження (x, y).
    best_k (list): Список індексів внутрішніх точок для кращої моделі.
    """
    # Ініціалізація кращої точки сходження та внутрішніх точок
    best_foe = (0, 0)
    best_inliers = 0
    best_k = []

    for _ in range(num_iterations):
        # Перевірка на мінімальну кількість точок
        if len(flow_vectors) < 2:
            continue

        # Випадковий вибір двох точок для розрахунку FOE
        indices = np.random.choice(len(flow_vectors), 2, replace=False)
        vector1, vector2 = flow_vectors[indices[0]], flow_vectors[indices[1]]

        # Розрахунок коефіцієнтів для лінії (FOE)
        x1, y1 = vector1
        x2, y2 = vector2
        a, b, c = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

        # Розрахунок кількості внутрішніх точок
        inliers = 0
        inlier_indices = []
        for i, (x, y) in enumerate(flow_vectors):
            if abs(a * x + b * y + c) < inlier_threshold:
                inliers += 1
                inlier_indices.append(i)

        # Оновлення кращої моделі, якщо знайдено більше внутрішніх точок
        if inliers > best_inliers:
            best_foe = (-c / (a + 1e-10), -c / (b + 1e-10))  # Уникнення ділення на нуль
            best_inliers = inliers
            best_k = inlier_indices

        # Якщо виявлено достатньо внутрішніх точок (поріг 90%), припинити ітерації
        if best_inliers > 0.9 * len(flow_vectors):
            break

    return best_foe, best_k

def inlier_static(q2, d, S, inlier_threshold):
    """
    Визначає внутрішні точки для статичної сцени за допомогою методу RANSAC
    та фільтрації точок на основі їх відстані від середнього значення.

    Параметри:
    q2 (numpy.ndarray): Координати точок на зображенні.
    d (numpy.ndarray): Векторні напрямки для кожної точки.
    S (tuple): Розміри зображення у вигляді (висота, ширина).
    inlier_threshold (float): Поріг для визначення внутрішніх точок за допомогою RANSAC.

    Повертає:
    tuple: Відфільтровані координати точок, відфільтровані напрямки, а також точка сходження (FOE), коригована на розмір зображення.
    """
    # Оцінка точки сходження (FOE) за допомогою методу RANSAC
    foe, inliers = estimate_foe(d, inlier_threshold)
    
    # Фільтрація точок за допомогою внутрішніх індексів
    q2, d = q2[inliers], d[inliers]
    
    # Обчислення евклідової відстані для кожної точки
    a = np.linalg.norm(d, axis=1)
    
    # Обчислення медіани відстаней
    dm = np.median(a)
    
    # Вибір точок, відстань яких менша за подвоєну медіану
    inlier_indices = np.where(a < 2 * dm)
    
    # Отримання точок, які відповідають фільтру
    qr, dr = q2[inlier_indices], d[inlier_indices]
    
    # Повернення відфільтрованих точок, напрямків та точки сходження з коригуванням на розмір зображення
    return qr, dr, foe + np.array([S[1] / 2, S[0] / 2])  # FOE коригується на розміри зображення
