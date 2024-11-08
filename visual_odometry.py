
import os
import numpy as np
import cv2
from tqdm import tqdm
from lib.lucas_kanade_flow import lucas_kanade_optical_flow
from lib.imu_integration import load_optitrak_data

class VisualOdometry:
    def __init__(self, data_directory):
        """
        Ініціалізує об'єкт VisualOdometry та завантажує необхідні дані з вказаного каталогу.

        Параметри:
        data_directory (str): Шлях до каталогу, що містить файли для калібрування, ground truth, IMU та зображення.
        """
        self.intrinsic_matrix, self.projection_matrix = self._load_calibration(os.path.join(data_directory, "calib.txt"))
        self.ground_truth_poses, self.imu_distances = self._load_ground_truth_and_imu(
            [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.startswith("ground_truth")],
            [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.startswith("imu")]
        )
        self.image_sequences = self._load_images(os.path.join(data_directory, "images"))

    @staticmethod
    def _load_calibration(filepath):
        """
        Завантажує калібрувальні параметри з вказаного файлу.

        Параметри:
        filepath (str): Шлях до файлу, що містить калібрувальні параметри.

        Повертає:
        tuple: Внутрішню матрицю та проекційну матрицю.
        """
        with open(filepath, 'r') as file:
            params = np.fromstring(file.readline(), dtype=np.float64, sep=' ')
            projection_matrix = np.reshape(params, (3, 4))
            intrinsic_matrix = projection_matrix[0:3, 0:3]
        return intrinsic_matrix, projection_matrix

    @staticmethod
    def _load_ground_truth_and_imu(filepath, timestamp):
        """
        Завантажує дані про позу та відстані з файлів OptiTrak та IMU.

        Параметри:
        filepath (list): Список шляхів до файлів з даними про позу.
        timestamp (list): Список шляхів до файлів з часовими мітками.

        Повертає:
        tuple: Трансформаційні матриці поз та масив відстаней.
        """
        poses, distances = load_optitrak_data(filepath[0], timestamp[0])
        return poses, distances

    @staticmethod
    def _load_images(filepath):
        """
        Завантажує зображення з вказаного каталогу.

        Параметри:
        filepath (str): Шлях до каталогу, який містить зображення.

        Повертає:
        list: Список зображень, завантажених за допомогою OpenCV.
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path) for path in image_paths]

    @staticmethod
    def _create_transformation_matrix(R, t):
        """
        Створює матрицю трансформації на основі матриці обертання та вектора переміщення.

        Параметри:
        R (ndarray): Матриця обертання розміру (3, 3).
        t (ndarray): Вектор переміщення розміру (3,).

        Повертає:
        ndarray: 4x4 матриця трансформації, яка комбінує обертання та переміщення.
        """
        transformation_matrix = np.eye(4, dtype=np.float64)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t
        return transformation_matrix

    def get_feature_matches(self, index, step, num_features=200, window_size=5, pyramid_level=5, num_iterations=70, inlier_threshold=3):
        """
        Знаходить відповідні точки між двома зображеннями за допомогою методу оптичного потоку Лукас-Канаде.

        Параметри:
        index (int): Поточний індекс зображення в послідовності.
        step (int): Крок між поточним і наступним зображенням для порівняння.
        num_features (int): Кількість характеристик для відслідковування.
        window_size (int): Розмір вікна для алгоритму Лукас-Канаде.
        pyramid_level (int): Кількість рівнів віртуальної піраміди.
        num_iterations (int): Кількість ітерацій для оцінки оптичного потоку.
        inlier_threshold (float): Поріг для визначення коректних точок.

        Повертає:
        tuple: Два масиви точок (q1, q2) з координатами відповідних точок на двох зображеннях.
        """
        q1, q2, focus_of_expansion = lucas_kanade_optical_flow(
            self.image_sequences[index - step], 
            self.image_sequences[index + 1 - step], 
            num_features, 
            window_size, 
            pyramid_level, 
            num_iterations, 
            inlier_threshold, 
        )
        image = self.image_sequences[index - step].copy()
        self._plot_optical_flow(image, q1, q2, focus_of_expansion)
        return q1, q2

    def compute_pose(self, q1, q2):
        """
        Обчислює матрицю обертання та вектор переміщення між двома зображеннями
        за допомогою есеціальної матриці.

        Параметри:
        q1 (array): Масив координат точок на першому зображенні.
        q2 (array): Масив координат точок на другому зображенні.

        Повертає:
        tuple: Матриця обертання (R) та вектор переміщення (t).
        """
        essential_matrix, _ = cv2.findEssentialMat(q1, q2, self.intrinsic_matrix, method=0, threshold=0.1)
        R, t = self._decompose_essential_matrix(essential_matrix, q1, q2)
        return R, t

    def _decompose_essential_matrix(self, essential_matrix, q1, q2):
        """
        Декомпозує есеціальну матрицю на матрицю обертання (R) та вектор переміщення (t),
        вибираючи найкращу комбінацію на основі кількості точок з позитивною координатою Z після триангуляції.

        Параметри:
        essential_matrix (array): Есеціальна матриця для декомпозиції.
        q1 (array): Масив координат точок на першому зображенні.
        q2 (array): Масив координат точок на другому зображенні.

        Повертає:
        tuple: Найкраща матриця обертання (R) та вектор переміщення (t).
        """
        def compute_positive_z_count(R, t):
            """
            Обчислює кількість точок з позитивною координатою Z після триангуляції
            для заданих матриці обертання (R) та вектора переміщення (t).

            Параметри:
            R (array): Матриця обертання.
            t (array): Вектор переміщення.

            Повертає:
            int: Загальна кількість точок з позитивною координатою Z після триангуляції.
            """
            transformation_matrix = self._create_transformation_matrix(R, t)
            projection_matrix = np.matmul(np.concatenate((self.intrinsic_matrix, np.zeros((3, 1))), axis=1), transformation_matrix)
            homogeneous_Q1 = cv2.triangulatePoints(np.float32(self.projection_matrix), np.float32(projection_matrix), np.float32(q1.T), np.float32(q2.T))
            homogeneous_Q2 = np.matmul(transformation_matrix, homogeneous_Q1)
            un_homogeneous_Q1 = homogeneous_Q1[:3, :] / homogeneous_Q1[3, :]
            un_homogeneous_Q2 = homogeneous_Q2[:3, :] / homogeneous_Q2[3, :]

            count_positive_z_Q1 = np.sum(un_homogeneous_Q1[2, :] > 0)
            count_positive_z_Q2 = np.sum(un_homogeneous_Q2[2, :] > 0)

            return count_positive_z_Q1 + count_positive_z_Q2

        R1, R2, t = cv2.decomposeEssentialMat(essential_matrix)
        t = np.squeeze(t)

        pairs = [[R1, -t], [R1, t], [R2, t], [R2, -t]]

        positive_z_counts = [compute_positive_z_count(R, t) for R, t in pairs]

        best_pair_index = np.argmax(positive_z_counts)
        best_R, best_t = pairs[best_pair_index]

        return best_R, best_t

    def _plot_optical_flow(self, image, points1, points2, focus_of_expansion):
        """
        Візуалізує оптичний потік між двома наборами точок на зображенні у вигляді стрілок та фокусу розширення.

        Параметри:
        image (array): Зображення, на якому відображатиметься оптичний потік.
        points1 (array): Масив точок на першому зображенні.
        points2 (array): Масив точок на другому зображенні.
        focus_of_expansion (tuple): Координати фокусу розширення (точка, де потік рухається).
        """
        features = np.intp(points1)
        depths = np.intp(points2)

        for i in range(len(depths)):
            cv2.arrowedLine(image, (features[i, 0], features[i, 1]), (depths[i, 0], depths[i, 1]), [255, 150, 0], 1, tipLength=0.2)

        cv2.circle(image, (np.intp(focus_of_expansion[0]), np.intp(focus_of_expansion[1])), 7, (0, 0, 255), -1)
        cv2.imshow('Optical Flow Frame', image)
        cv2.waitKey(10)

def main(queued_data, data_directory, gt_path_3d, estimated_path_3d, num_features=200, window_size=5, pyramid_level=5, num_iterations=70, inlier_threshold=3):
    """
    Головна функція для виконання візуальної одометрії. 
    Функція ініціалізує об'єкт візуальної одометрії, обробляє кадри з даними про позу та IMU, отримує відповідні точки за допомогою оптичного потоку Лукас-Канаде,
    а також обчислює трансформаційні матриці для кожного кадру. Крім того, зберігаються та оновлюються орієнтації і пози для оціненого та істинного шляху.
    
    Параметри:
    queued_data (queue): Черга для обміну даними між процесами.
    data_directory (str): Шлях до каталогу з вхідними даними.
    gt_path_3d (list): Список для зберігання оціненого 3D шляху.
    estimated_path_3d (list): Список для зберігання оцінених 3D шляхів.
    num_features (int): Кількість особливостей для пошуку (за замовчуванням 200).
    window_size (int): Розмір вікна для оптичного потоку (за замовчуванням 5).
    pyramid_level (int): Кількість рівнів піраміди для оптичного потоку (за замовчуванням 5).
    num_iterations (int): Кількість ітерацій для оптичного потоку (за замовчуванням 70).
    inlier_threshold (int): Поріг для верифікації інлайнів (за замовчуванням 3).
    is_static (bool): Прапорець для визначення, чи є сцена статичною (за замовчуванням False).
    """
    # Ініціалізація об'єкта VisualOdometry
    vo = VisualOdometry(data_directory)
    imu_distances = vo.imu_distances
    ground_truth_poses = vo.ground_truth_poses
    current_poses = np.copy(ground_truth_poses)
    
    start_frame = 1  # Початковий кадр для обробки
    step = 1  # Крок
    end_frame = 5  # Кількість кадрів для пропуску на кінці

    # Обрізаємо позу для початкових кадрів
    current_poses = np.array(current_poses)[start_frame:-(end_frame):step, :, :]
    
    estimated_rotations = []
    ground_truth_rotations = []
    current_pose = current_poses[0]

    # Проходимо через усі кадри
    for i, gt_pose in zip(range(start_frame, len(current_poses) - end_frame, step), tqdm(current_poses, unit=" Frames", smoothing=0, disable=True)):
        shared_data = [i - start_frame, len(current_poses), 1]
        queued_data.put(shared_data)

        if i > start_frame:
            # Отримуємо відповідні точки для оптичного потоку
            q1, q2 = vo.get_feature_matches(i, step, num_features, window_size, pyramid_level, num_iterations, inlier_threshold)
            R, t = vo.compute_pose(q1, q2)
            t *= imu_distances[i]
            transformation = vo._create_transformation_matrix(R, np.squeeze(t))
            current_pose = np.matmul(current_pose, np.linalg.inv(transformation))

        # Зберігаємо обертання та пози
        gt_rotation, _ = cv2.Rodrigues(gt_pose[:3, :3])
        estimated_rotation, _ = cv2.Rodrigues(current_pose[:3, :3])
        ground_truth_rotations.append(gt_rotation)
        estimated_rotations.append(estimated_rotation)
        estimated_path_3d.append((estimated_rotation[0][0], estimated_rotation[1][0], estimated_rotation[2][0], current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]))
        gt_path_3d.append((gt_rotation[0][0], gt_rotation[1][0], gt_rotation[2][0], gt_pose[0, 3], gt_pose[1, 3], gt_pose[2, 3]))

if __name__ == "__main__":
    main()
