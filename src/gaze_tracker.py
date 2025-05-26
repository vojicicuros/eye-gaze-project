import os
import threading
import time
import joblib
from .camera_feed import Camera
from .landmark_detector import Detector
from sklearn.linear_model import LinearRegression
import numpy as np
import json
from screeninfo import get_monitors


# constants
black = (0, 0, 0)
red = (255, 0, 0)
white = (255, 255, 255)
radius = 20
padding = 50
transition_steps = 15
transition_time = 0.02
collapse_steps = 20
collapse_time = 0.05
num_of_dots = 3  # 3x3


class GazeTracker:
    def __init__(self):
        self.env_cleanup()
        self.cam = Camera()
        self.detector = Detector(camera=self.cam)

        self.screen_height = None
        self.screen_width = None

        # Initialize the model
        #self.model = self.import_model()

        from gui.calibration import Calibration
        from gui.validation import Validation

        self.calibration_data = self.read_from_file(filename="iris_data_fix.json")
        self.gaze = None

        self.screen_positions = self.calculate_positions(num_of_dots)
        self.calibration = Calibration(self)
        self.validation = Validation(self)

        # self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.calibration_data_thread = threading.Thread(target=self.calibration_iris_data_to_file)
        self.validation_data_thread = threading.Thread(target=self.validation_iris_data)
        self.exit_event = threading.Event()

    def read_from_file(self, filename):
        file_path = os.path.join("data", filename)
        print(f"Reading from: {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                print("Successfully loaded data:")
                return data

        except Exception as e:
            print(f"Unexpected error while reading the file: {e}")
            return None

    def normalize(self, point, frame_width, frame_height):
        return [point[0] / frame_width, point[1] / frame_height]

    def linear_estimation(self, live_data):
        print(f"Live data: {live_data}")

        for key in [1, 3, 5, 7]:
            print(f"[{key}] l_iris_center: {self.calibration_data[key]['l_iris_center']}")
            print(f"[{key}] r_iris_center: {self.calibration_data[key]['r_iris_center']}")
            print(f"[{key}] screen_position: {self.calibration_data[key]['screen_position']}")

        x, y = live_data

        x1 = np.average([self.calibration_data[3]['l_iris_center'],
                         self.calibration_data[3]['r_iris_center']], axis=0)[0]
        x2 = np.average([self.calibration_data[5]['l_iris_center'],
                         self.calibration_data[5]['r_iris_center']], axis=0)[0]

        alpha1 = self.calibration_data[3]['screen_position'][0]
        alpha2 = self.calibration_data[5]['screen_position'][0]

        y1 = np.average([self.calibration_data[1]['l_iris_center'],
                         self.calibration_data[1]['r_iris_center']], axis=0)[1]
        y2 = np.average([self.calibration_data[7]['l_iris_center'],
                         self.calibration_data[7]['r_iris_center']], axis=0)[1]

        beta1 = self.calibration_data[1]['screen_position'][1]
        beta2 = self.calibration_data[7]['screen_position'][1]

        alpha = alpha1 + (x - x1) / (x2 - x1) * (alpha2 - alpha1)
        beta = beta1 + (y - y1) / (y2 - y1) * (beta2 - beta1)

        return np.array([alpha, beta])

    def calibration_iris_data(self):

        iris_data_dict = {
            "l_iris_center": [],
            "r_iris_center": [],
            "avg_center": [],
            "screen_position": []
        }
        l_iris_data = []
        r_iris_data = []
        was_collecting = False  # Tracks the previous state of the flag

        while not self.calibration.exit_event.is_set():
            if self.calibration.iris_data_flag:
                was_collecting = True
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")
                r_iris_cent = self.detector.camera.eyes_landmarks.get("r_iris_center")

                l_iris_data.append(l_iris_cent)
                r_iris_data.append(r_iris_cent)

            else:
                # Only append once when data_flag switches from True to False
                if was_collecting:
                    l_median = np.median(l_iris_data, axis=0).astype(int).tolist()
                    r_median = np.median(r_iris_data, axis=0).astype(int).tolist()
                    iris_data_dict["l_iris_center"].append(l_median)
                    iris_data_dict["r_iris_center"].append(r_median)
                    iris_data_dict["avg_center"].append(np.average([l_median, r_median], axis=0).tolist())

                    l_iris_data = []
                    r_iris_data = []
                    was_collecting = False  # Reset flag to prevent repeated appending
            time.sleep(0.01)

        iris_data_dict["screen_position"] = self.screen_positions[1:]

        iris_data_list = []
        for l_iris, r_iris, avg, screen_pos in zip(iris_data_dict["l_iris_center"],
                                                   iris_data_dict["r_iris_center"],
                                                   iris_data_dict["avg_center"],
                                                   self.screen_positions[1:]):
            iris_data_list.append({"l_iris_center": l_iris,
                                   "r_iris_center": r_iris,
                                   "avg_center": avg,
                                   "screen_position": screen_pos})

        return iris_data_list

    def validation_iris_data(self):

        while not self.validation.exit_event.is_set():
            if self.validation.iris_data_flag:
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")
                r_iris_cent = self.detector.camera.eyes_landmarks.get("r_iris_center")
                avg_iris = np.average([l_iris_cent, r_iris_cent], axis=0)
                #print(avg_iris)
                self.gaze = self.linear_estimation(avg_iris)
                print(self.gaze)

            time.sleep(0.01)

    def calibration_iris_data_to_file(self):

        iris_data = self.calibration_iris_data()
        self.save_data_to_file(data=iris_data, filename="iris_data.json")

    def save_data_to_file(self, data, filename):
        file_path = os.path.join("data", filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            print(f"Saved data into f{filename} file.")

    def env_cleanup(self):
        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")

    def calculate_positions(self, n=num_of_dots):
        monitor = get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height

        row_step = (screen_height - 2 * padding) // (n-1)
        col_step = (screen_width - 2 * padding) // (n-1)

        positions = [(screen_width // 2, screen_height // 2)] + [
            (padding + i * col_step, padding + j * row_step)
            for j in range(n) for i in range(n)
        ]
        return positions

    def import_model(self):
        model_path = os.path.join("data", "linear_model.joblib")
        if os.path.exists(model_path):
            print(f"Loading pretrained model from {model_path}.")
            return joblib.load(model_path)
        else:
            print("No pretrained model found. Using a new one.")
            return LinearRegression()



