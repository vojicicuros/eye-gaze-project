import os
import threading
import time
from .camera_feed import Camera
from .landmark_detector import Detector
import numpy as np
import json
from screeninfo import get_monitors
from consts import *


class GazeTracker:
    def __init__(self):
        self.env_cleanup()
        self.cam = Camera()
        self.detector = Detector(camera=self.cam)

        self.screen_height = screen_height
        self.screen_width = screen_width

        print(f'Window size {self.screen_width}x{self.screen_height}')

        from gui.calibration import Calibration
        from gui.validation import Validation
        from gui.gaze_part import Gazing

        self.x1 = 0
        self.x2 = 0
        self.alpha1 = 0
        self.alpha2 = 0
        self.y1 = 0
        self.y2 = 0
        self.beta1 = 0
        self.beta2 = 0

        self.calibration_data = None
        self.gaze = None

        self.screen_positions = self.calculate_positions(num_of_dots)
        self.calibration = Calibration(self)
        self.validation = Validation(self)
        self.gazing_part = Gazing(self)

        # self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.calibration_data_thread = threading.Thread(target=self.calibration_iris_data_wrap)
        self.validation_data_thread = threading.Thread(target=self.validation_iris_data)
        self.exit_event = threading.Event()

    def read_from_file(self, filename):
        file_path = os.path.join("data", filename)
        print(f"Reading from: {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                print("Successfully loaded data")
                return data

        except Exception as e:
            print(f"Unexpected error while reading the file: {e}")
            return None


    def linear_estimation(self, live_data):

        x, y = live_data

        # Clamp the x and y to be within the calibration bounds
        x = np.clip(x, self.x1, self.x2)
        y = np.clip(y, self.y1, self.y2)

        # Linear interpolation
        alpha = self.alpha1 + (x - self.x1) / (self.x2 - self.x1) * (self.alpha2 - self.alpha1)
        beta = self.beta1 + (y - self.y1) / (self.y2 - self.y1) * (self.beta2 - self.beta1)

        # Optional: clamp again to ensure you're within screen size
        alpha = np.clip(alpha, padding, self.screen_width - padding)
        alpha = np.abs(self.screen_width - padding - alpha)
        beta = np.clip(beta, padding, self.screen_height - padding)

        return np.array([alpha, beta])

    def calculate_consts(self):

        # Horizontal calibration
        self.x1 = round(np.mean([self.calibration_data[0]['l_iris_center'][0],
                                 self.calibration_data[3]['l_iris_center'][0],
                                 self.calibration_data[6]['l_iris_center'][0]]))
        self.x2 = round(np.mean([self.calibration_data[2]['l_iris_center'][0],
                                 self.calibration_data[5]['l_iris_center'][0],
                                 self.calibration_data[8]['l_iris_center'][0]]))
        self.alpha1 = self.calibration_data[3]['screen_position'][0]
        self.alpha2 = self.calibration_data[5]['screen_position'][0]

        if self.x1 > self.x2:
            self.x1, self.x2 = self.x2, self.x1

        # Vertical calibration
        self.y1 = round(np.mean([self.calibration_data[0]['l_iris_center'][1],
                                 self.calibration_data[1]['l_iris_center'][1],
                                 self.calibration_data[2]['l_iris_center'][1]]))
        self.y2 = round(np.mean([self.calibration_data[6]['l_iris_center'][1],
                                 self.calibration_data[7]['l_iris_center'][1],
                                 self.calibration_data[8]['l_iris_center'][1]]))
        self.beta1 = self.calibration_data[1]['screen_position'][1]
        self.beta2 = self.calibration_data[7]['screen_position'][1]

    def calibration_iris_data(self):

        iris_data_dict = {
            "l_iris_center": [],
            # "r_iris_center": [],
            # "avg_center": [],
            "screen_position": []
        }
        l_iris_data = []
        r_iris_data = []
        was_collecting = False  # Tracks the previous state of the flag

        while not self.calibration.exit_event.is_set():
            if self.calibration.iris_data_flag:
                was_collecting = True
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")
                # r_iris_cent = self.detector.camera.eyes_landmarks.get("r_iris_center")

                l_iris_data.append(l_iris_cent)
                # r_iris_data.append(r_iris_cent)

            else:
                # Only append once when data_flag switches from True to False
                if was_collecting:
                    l_median = np.median(l_iris_data, axis=0).astype(int).tolist()
                    # r_median = np.median(r_iris_data, axis=0).astype(int).tolist()
                    iris_data_dict["l_iris_center"].append(l_median)
                    # iris_data_dict["r_iris_center"].append(r_median)
                    # iris_data_dict["avg_center"].append(np.average([l_median, r_median], axis=0).tolist())

                    l_iris_data = []
                    r_iris_data = []
                    was_collecting = False  # Reset flag to prevent repeated appending
            time.sleep(0.01)

        iris_data_dict["screen_position"] = self.screen_positions[1:]

        iris_data_list = []
        for l_iris, screen_pos in zip(iris_data_dict["l_iris_center"],
                                      # iris_data_dict["r_iris_center"],
                                      # iris_data_dict["avg_center"],
                                      self.screen_positions[1:]):
            iris_data_list.append({"l_iris_center": l_iris,
                                   # "r_iris_center": r_iris,
                                   # "avg_center": avg,
                                   "screen_position": screen_pos})

        return iris_data_list

    def validation_iris_data(self):

        self.calibration_data = self.read_from_file(filename=filename)
        self.calculate_consts()

        while not self.validation.exit_event.is_set():
            if self.validation.iris_data_flag:
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")
                # r_iris_cent = self.detector.camera.eyes_landmarks.get("r_iris_center")
                # avg_iris = np.average([l_iris_cent, r_iris_cent], axis=0)
                self.gaze = self.linear_estimation(l_iris_cent)

            time.sleep(0.01)

    def calibration_iris_data_wrap(self):

        iris_data = self.calibration_iris_data()
        self.save_data_to_file(data=iris_data)

    def save_data_to_file(self, data):

        file_path = os.path.join("data", filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            print(f"Saved data into {filename} file.")

    def env_cleanup(self):

        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")

    def calculate_positions(self, n=num_of_dots):

        row_step = (screen_height - 2 * padding) // (n-1)
        col_step = (screen_width - 2 * padding) // (n-1)

        positions = [(screen_width // 2, screen_height // 2)] + [
            (padding + i * col_step, padding + j * row_step)
            for j in range(n) for i in range(n)
        ]
        return positions



