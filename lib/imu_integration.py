import csv
import numpy as np
import cv2
from scipy.integrate import cumulative_trapezoid

def load_optitrak_data(filename, timestamps):
    """
    Завантажує дані з файлів OptiTrak і обробляє їх для створення трансформаційних матриць.

    Параметри:
    filename (str): Шлях до файлу з даними OptiTrak.
    timestamps (str): Шлях до файлу з часовими мітками.

    Повертає:
    tuple: Список трансформаційних матриць і масив відстаней.
    """
    timestamp = []
    
    # Завантаження часових міток з файлу
    with open(timestamps, newline='\n', encoding='utf-8') as csvfile:
        file = csv.reader(csvfile, delimiter=',', quotechar='|')
        k = 0
        for time_stamp in file:
            if k > 0:
                timestamp.append((time_stamp[1], time_stamp[2], time_stamp[6], time_stamp[7], 
                                  time_stamp[8], time_stamp[9], time_stamp[10], time_stamp[11], time_stamp[12]))
            k += 1
    timestamp = np.array(timestamp, dtype=np.float64)
    timestamp[:,-2] += 981  # Коригуємо координати 
    poses = []
    j = 0
    i = 0

    # Завантажуємо дані позицій з файлу
    with open(filename, newline='\n', encoding='utf-8') as csvfile:
        rec = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rec:
            if i >= 7 and j < len(timestamp):
                # Пошук найближчої часової мітки
                if abs(float(row[1]) - float(timestamp[j, 0])) < 0.01:
                    if '' not in row[2:8]:  # Перевірка на порожні значення
                        row = np.array(row[2:8], dtype=np.float32)
                        x, y, z, t_x, t_y, t_z = row
                        
                        # Перетворення даних на радіани для обертання
                        optitrak_rotation = np.array([np.radians(x), np.radians(y), np.radians(z)])
                        optitrak_translation_vector = [t_x, t_y, t_z]

                        # Обчислення матриці обертання і вектору трансляції
                        rotation_matrix_opencv, _ = cv2.Rodrigues(optitrak_rotation)
                        translation_vector_opencv = np.array(optitrak_translation_vector)

                        # Створення коригуючої матриці для орієнтації
                        R = np.array([[0,0,1], [0,1,0], [1,0,0]])
                        transformation_matrix_opencv = np.eye(4)
                        transformation_matrix_opencv[:3, :3] = rotation_matrix_opencv
                        transformation_matrix_opencv[:3, 3] = translation_vector_opencv
                        transformation_matrix_opencv[:3, :] = np.matmul(R, transformation_matrix_opencv[:3, :])

                        # Додаємо отриману матрицю в список
                        poses.append(transformation_matrix_opencv)
                        
                        j += 1
            i += 1  # Крок індексу рядка

    # Обчислення відстані
    dist = integrate((timestamp[:, 2:8]) / 100, timestamp[:, 0])
    
    return poses, dist

def integrate(a, t):
    """
    Обчислює відстань на основі прискорень і часових міток за допомогою чисельного інтегрування.

    Параметри:
    a (numpy.ndarray): Масив прискорень для кожної осі.
    t (numpy.ndarray): Часові мітки.

    Повертає:
    list: Список відстаней між точками.
    """
    # Розрахунок швидкості по осях X, Y, Z
    vx = cumulative_trapezoid(a[:, 3] + a[:, 4], t, initial=0)  # Швидкість по X
    vy = cumulative_trapezoid(a[:, 4] - a[:, 4], t, initial=0)  # Швидкість по Y
    vz = cumulative_trapezoid(-a[:, 5] - a[:, 4], t, initial=0) # Швидкість по Z

    # Розрахунок переміщення по кожній осі
    dx = cumulative_trapezoid(vx, t, initial=0)
    dy = cumulative_trapezoid(vy, t, initial=0)
    dz = cumulative_trapezoid(vz, t, initial=0)

    # Створення масиву швидкостей
    vo = np.zeros((len(dx), 3))
    vo[:, 0], vo[:, 1], vo[:, 2] = dx, dy, dz

    # Обчислення відстані між точками
    d_vo = []
    for i in range(1, len(vo)):
        d_vo.append(np.linalg.norm(vo[i] - vo[i - 1]))  # Евклідова відстань між точками

    return d_vo
