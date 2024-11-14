import cv2
import numpy as np
from lib.foe_ransac import inlier_static

def calculate_window_indices(points, window_size, image_shape, level):

    x_range = [np.arange(val - window_size, val + window_size + 1) for val in points[:, 0]]
    y_range = [np.arange(val - window_size, val + window_size + 1) for val in points[:, 1]]

    # Обмеження індексів з урахуванням розмірів зображення
    x = np.minimum(np.array(x_range), image_shape[level, 1] - 1)
    y = np.minimum(np.array(y_range), image_shape[level, 0] - 1)

    # Обмеження індексів до 0
    x[x < 0] = 0
    y[y < 0] = 0

    # Створення комбінованих індексів для кожного пікселя
    x_combined = np.repeat(x, window_size, axis=1)
    y_combined = np.tile(y, window_size)

    # Пошук попередніх індексів
    x_prev = np.maximum(x_combined - 1, 0)
    y_prev = np.maximum(y_combined - 1, 0)

    return np.intp(x_combined), np.intp(y_combined), np.intp(x_prev), np.intp(y_prev)


def calculate_gradients(image, window_shape, level, x, y, x_prev, y_prev):

    I_x = (image[y, np.minimum(x + 1, window_shape[level, 1] - 1)] - image[y, x_prev]) / 2
    I_y = (image[np.minimum(y + 1, window_shape[level, 0] - 1), x] - image[y_prev, x]) / 2
    return I_x, I_y


def compute_intensity_difference(I_L, J_L, x, y, velocity, gradient, level, shape):

    vy = velocity[:, 1].reshape(len(velocity), 1)
    vx = velocity[:, 0].reshape(len(velocity), 1)
    gy = gradient[:, 1].reshape(len(gradient), 1)
    gx = gradient[:, 0].reshape(len(gradient), 1)

    k = np.clip(np.intp(np.round(y + vy + gy)), 0, shape[level, 0] - 1)
    m = np.clip(np.intp(np.round(x + vx + gx)), 0, shape[level, 1] - 1)

    intensity_diff = I_L[y, x] - J_L[k, m]
    return intensity_diff


def lucas_kanade_optical_flow(img1_, img2_, number_features, window_size, pyramid_levels, iterations, inlier_threshold):

    # Перетворення в відтінки сірого
    img1 = cv2.cvtColor(img1_, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_, cv2.COLOR_BGR2GRAY)

    # Створення пірамід зображень
    I_L = np.empty((img1.shape[0], img1.shape[1], pyramid_levels), dtype=np.float32)
    J_L = np.empty((img1.shape[0], img1.shape[1], pyramid_levels), dtype=np.float32)
    I_L[:, :, 0] = img1
    J_L[:, :, 0] = img2

    shape = np.empty((pyramid_levels, 2), dtype=int)
    shape[0, :] = img1.shape

    # Створення рівнів піраміди
    for i in range(1, pyramid_levels):
        shape[i, :] = np.shape(cv2.resize(I_L[0:shape[i-1, 0], 0:shape[i-1, 1], i-1], None, fx=0.5, fy=0.5))
        I_L[0:shape[i, 0], 0:shape[i, 1], i] = cv2.resize(I_L[0:shape[i-1, 0], 0:shape[i-1, 1], i-1], None, fx=0.5, fy=0.5)
        J_L[0:shape[i, 0], 0:shape[i, 1], i] = cv2.resize(J_L[0:shape[i-1, 0], 0:shape[i-1, 1], i-1], None, fx=0.5, fy=0.5)

    # Знаходимо риси для відстеження
    features = cv2.goodFeaturesToTrack(I_L[:, :, 0], number_features, 0.01, 10)
    points = np.intp(np.array(features).reshape(len(features), -1))

    # Ініціалізація швидкостей
    g_Lm = np.zeros((len(points), 2), dtype=np.float32)
    velocities = []

    # Оптичний потік для кожного рівня
    for level in range(pyramid_levels - 1, -1, -1):
        q = np.intp(points / (2 ** level))
        x, y, x_prev, y_prev = calculate_window_indices(q, window_size, shape, level)
        I_x, I_y = calculate_gradients(I_L[:, :, level], shape, level, x, y, x_prev, y_prev)

        # Обчислення матриці градієнтів
        I2x = I_x * I_x
        I2y = I_y * I_y
        Ixy = I_x * I_y

        I_x2_sum = np.sum(I2x, axis=1)
        I_y2_sum = np.sum(I2y, axis=1)
        I_xy_sum = np.sum(Ixy, axis=1)

        G = np.dstack((I_x2_sum, I_xy_sum, I_xy_sum, I_y2_sum)).reshape(len(I_x), 2, 2)
        G_inv = np.linalg.pinv(G)

        # Ініціалізація швидкості
        velocity = np.zeros((len(x), 2), dtype=np.float32)

        # Виконання ітерацій для поліпшення результату
        for _ in range(iterations):
            dIk = compute_intensity_difference(I_L[:, :, level], J_L[:, :, level], x, y, velocity, g_Lm, level, shape)
            dIk_x = dIk * I_x
            dIk_y = dIk * I_y
            b = np.sum(np.dstack((dIk_x, dIk_y)), axis=1).reshape(len(I_x), 2, 1)

            # Обчислення оптичного потоку за методом Лукас-Канаде
            delta_v = np.matmul(G_inv, b).reshape(velocity.shape)
            velocity += delta_v

        g_Lm = 2 * (velocity + g_Lm)
        velocities.append(np.median(velocity, axis=0))

    # Розрахунок остаточного руху
    final_displacement = g_Lm / 2
    points, final_displacement, outliers = inlier_static(points, final_displacement, img1.shape, inlier_threshold)

    # Обчислення нових координат
    new_points = points + final_displacement
    return points, new_points, outliers
