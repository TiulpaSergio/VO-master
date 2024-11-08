import tkinter as tk
import cv2
from scipy.spatial.transform import Rotation
from tkinter import ttk
import numpy as np
import queue
import threading
import matplotlib.pyplot as plt
import visual_odometry as visual_odometry  
from mpl_toolkits.mplot3d import Axes3D

class VOApp:
    def __init__(self, root):
        """
        Ініціалізація кореневого вікна додатку.

        Параметри:
        root (Tk): Кореневий елемент вікна додатку.
        """
        self.root = root
        self.root.title("Visual Odometry")

        # Початкові значення параметрів
        self.initial_param1 = 500  # кількість ознак
        self.initial_param2 = 70   # кількість ітерацій
        self.initial_param3 = 3    # поріг вхідних точок

        # Константи для розміру вікна та рівня піраміди
        self.optical_flow_window_size = 5  # Константа розміру вікна для оптичного потоку
        self.pyramid_level = 5  # Константа рівня піраміди

        # Створення елементів інтерфейсу
        self.setup_ui()

    def setup_ui(self):
        """
        Створює графічний інтерфейс користувача для налаштування параметрів візуальної одометрії.

        Цей метод ініціалізує фрейм для введення параметрів, створює мітки та поля вводу для кожного 
        параметра, включаючи кількість ознак, кількість ітерацій та поріг вхідних точок. 
        Також додається кнопка "Старт" для запуску процесу візуальної одометрії.
        """

        # Створення фрейму для вибору параметрів (по центру по горизонталі)
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side="top", padx=20, pady=20, fill='y', expand=True)

        # Мітка для параметрів введення
        self.param_label = tk.Label(self.control_frame, text="Введіть параметри:")
        self.param_label.grid(row=0, column=0, pady=10, columnspan=2)

        # Створення полів для вводу параметрів
        self.param1_label = tk.Label(self.control_frame, text="Кількість ознак:")
        self.param1_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.param1_entry = tk.Entry(self.control_frame)
        self.param1_entry.insert(0, self.initial_param1)  # Початкове значення
        self.param1_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.param2_label = tk.Label(self.control_frame, text="Кількість ітерацій:")
        self.param2_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")
        self.param2_entry = tk.Entry(self.control_frame)
        self.param2_entry.insert(0, self.initial_param2)  # Початкове значення
        self.param2_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

        self.param3_label = tk.Label(self.control_frame, text="Поріг вхідних точок:")
        self.param3_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
        self.param3_entry = tk.Entry(self.control_frame)
        self.param3_entry.insert(0, self.initial_param3)  # Початкове значення
        self.param3_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # Створення кнопки "Старт"
        self.start_button = ttk.Button(self.control_frame, text="Старт", command=self.start_vo)
        self.start_button.grid(row=6, column=0, columnspan=2, pady=10)

        # Ініціалізація змінних для потоку
        self.vo_thread = None
        self.folder_path = "dataset"  # Шлях до папки з даними
        self.queue = queue.Queue()  # Черга для обміну даними між потоками

    def start_vo(self):
        """
        Запускає процес візуальної одометрії у новому потоці, використовуючи параметри, введені користувачем.

        Цей метод перевіряє, чи вже працює потік візуальної одометрії. Якщо потік не працює, то він ініціалізує 
        необхідні параметри, такі як кількість ознак, розмір вікна для оптичного потоку, рівень піраміди, кількість 
        ітерацій і поріг вхідних точок, отримані з полів вводу користувача. Потім він створює новий потік, в якому 
        виконується метод `run_vo`, передаючи йому ці параметри разом зі шляхом до даних і списками для збереження результатів.
        """

        # Перевірка, чи потік вже не працює
        if self.vo_thread is None or not self.vo_thread.is_alive():
            ground_truth = []  # Список для істинних значень
            monocular_vo = []  # Список для результатів візуальної одометрії
            param1 = int(self.param1_entry.get())  # Кількість ознак
            param2 = int(self.param2_entry.get())  # Кількість ітерацій
            param3 = float(self.param3_entry.get())  # Поріг вхідних точок
                           
            # Запуск візуальної одометрії в окремому потоці з параметрами
            self.vo_thread = threading.Thread(target=self.run_vo, args=(
                ground_truth, monocular_vo, param1, 
                self.optical_flow_window_size, self.pyramid_level, 
                param2, param3, self.folder_path
            ))
            self.vo_thread.start()

    def run_vo(self, ground_truth, monocular_vo, param1, optical_flow_window_size, pyramid_level, param2, param3, folder_path):
        """
        Виконує основний процес візуальної одометрії, обробляючи результати та візуалізуючи порівняння шляху візуальної одометрії 
        з істинними даними.

        Цей метод викликає основну функцію візуальної одометрії, передаючи їй необхідні параметри, включаючи список істинних значень 
        та результатів візуальної одометрії. Потім відбувається обробка результатів, включаючи ротацію шляху візуальної одометрії 
        для вирівнювання з істинними даними, застосування корекції через матрицю обертання, та візуалізація результатів на 3D графіку.

        Параметри:
        - ground_truth (list): Список істинних значень 3D координат (шлях).
        - monocular_vo (list): Список результатів візуальної одометрії (шлях).
        - param1 (int): Кількість ознак.
        - param2 (int): Кількість ітерацій.
        - param3 (float): Поріг вхідних точок.
        - folder_path (str): Шлях до папки з даними.

        """

        visual_odometry.main(self.queue, folder_path, ground_truth, monocular_vo, param1, optical_flow_window_size, pyramid_level, param2, param3)
        
        cv2.destroyAllWindows()
        
        gt_3d = np.array(ground_truth)
        gt_path_3d = gt_3d[:, 3:]
        gt_rot_vec = gt_3d[:, :3]
        estimated_3d = np.array(monocular_vo)
        estimated_path_3d = estimated_3d[:, 3:]
        estimated_rot_vec = estimated_3d[:, :3]
        vo_to_rotate = np.copy(estimated_path_3d - gt_path_3d[0])

        # Ротація шляху візуальної одометрії для вирівнювання з істинними даними
        rotate_matrix = []
        for i in range(len(estimated_rot_vec)):
            R, _ = cv2.Rodrigues(estimated_rot_vec[i] - gt_rot_vec[0]) 
            rotate_matrix.append(R)
        rotate_matrix = np.array(rotate_matrix)

        # Оборотна матриця для корекції
        hec_rotation_matrix = Rotation.from_euler('xyz', [0, 190, 170], degrees=True)
        hec_rotation_matrix = hec_rotation_matrix.as_matrix()
        vo_to_rotate = vo_to_rotate[:, :, np.newaxis]
        vo_rotated = hec_rotation_matrix @ vo_to_rotate
        estimated_path_3d = np.squeeze(vo_rotated, axis=2) 
        estimated_path_3d = np.copy(estimated_path_3d + gt_path_3d[0])

        # Візуалізація результатів на 3D графіку
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.plot(estimated_path_3d[:, 2], estimated_path_3d[:, 0], estimated_path_3d[:, 1], label='Visual Odometry')
        ax.plot(gt_path_3d[:, 2], gt_path_3d[:, 0], gt_path_3d[:, 1], label='Ground Truth')
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

if __name__ == "__main__":
    # Ініціалізація головного вікна та запуск додатку
    root = tk.Tk()
    app = VOApp(root)
    root.mainloop()
