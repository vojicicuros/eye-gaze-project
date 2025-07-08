import math
import os
import threading
import time
from .camera_feed import Camera
from .landmark_detector import Detector
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from consts import *
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class GazeTracker:
    def __init__(self):
        self.env_cleanup()
        self.cam = Camera()
        self.detector = Detector(camera=self.cam)

        self.screen_height = screen_height
        self.screen_width = screen_width

        from gui.calibration import Calibration
        from gui.validation import Validation
        from gui.gaze_part import Gazing

        self.svr_model_y = None
        self.svr_model_x = None
        self.poly_reg_y = None
        self.poly_reg_x = None

        self.x1 = self.x2 = self.alpha1 = self.alpha2 = 0
        self.y1 = self.y2 = self.beta1 = self.beta2 = 0

        self.calibration_data = None
        self.gaze = None

        self.all_metrics = []
        self.metrics_summary = None

        self.screen_positions = self.calculate_positions(num_of_dots)
        self.calibration = Calibration(self)
        self.validation = Validation(self)
        self.gazing_part = Gazing(self)

        # self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.calibration_data_thread = threading.Thread(target=self.calibration_iris_data_wrap)
        self.validation_data_thread = threading.Thread(target=self.validation_iris_data)
        self.exit_event = threading.Event()

    def read_from_file(self):
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

    def linear_mapping(self, live_data):

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

    def train_polynomial_regression(self):
        # 1. Extract features and labels
        X = []
        Yx = []
        Yy = []

        for entry in self.calibration_data:
            x, y = entry["l_iris_center"]
            sx, sy = entry["screen_position"]

            # Polynomial features: 1, x, y, x^2, y^2, xy
            X.append([1, x, y, x ** 2, y ** 2, x * y])
            Yx.append(sx)
            Yy.append(sy)

        X = np.array(X)
        Yx = np.array(Yx)
        Yy = np.array(Yy)

        print(X, Yx, Yy)

        # 2. Train two regressors
        self.poly_reg_x = LinearRegression().fit(X, Yx)
        self.poly_reg_y = LinearRegression().fit(X, Yy)

        print("Polynomial regression trained.")

    def train_svr(self):
        # Extract features and labels
        X = []
        Yx = []
        Yy = []

        for entry in self.calibration_data:
            x, y = entry["l_iris_center"]
            sx, sy = entry["screen_position"]

            # Same feature set used in linear regression (can be adjusted if needed)
            X.append([x, y])
            Yx.append(sx)
            Yy.append(sy)

        X = np.array(X)
        Yx = np.array(Yx)
        Yy = np.array(Yy)

        # Create and train SVR with standard scaling
        self.svr_model_x = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=500, epsilon=0.5))
        self.svr_model_y = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=500, epsilon=0.5))

        self.svr_model_x.fit(X, Yx)
        self.svr_model_y.fit(X, Yy)

        print("SVR model trained.")

    def polynomial_mapping(self, iris_center):
        """
        Predict the gaze point (screen coordinates) from iris center using polynomial regression.

        Args:
            iris_center: tuple or list of (x, y) representing iris center in camera coordinates.
        Returns:
            (screen_x, screen_y): predicted screen coordinates as floats.
        """
        if iris_center is None:
            return None

        x, y = iris_center
        features = np.array([[1, x, y, x ** 2, y ** 2, x * y]])

        try:
            alpha = self.poly_reg_x.predict(features)[0]
            beta = self.poly_reg_y.predict(features)[0]
            # Clamp values to stay within screen bounds with padding
            alpha = np.clip(alpha, padding, self.screen_width - padding)
            beta = np.clip(beta, padding, self.screen_height - padding)
            return np.array([alpha, beta])

        except Exception as e:
            print("Polynomial mapping error:", e)
            return None

    def svr_mapping(self, iris_center):
        """
            Predict the gaze point (screen coordinates) from the iris center using trained SVR models.

            Args:
                iris_center (tuple): A tuple (x, y) representing the iris center coordinates in camera space.

            Returns:
                (screen_x, screen_y): predicted screen coordinates as floats.

            Notes:
                - Uses two separate SVR models: one for horizontal (X) and one for vertical (Y) predictions.
                - Output is clamped to the screen boundaries using predefined padding.
            """
        if iris_center is None:
            return None

        x, y = iris_center
        features = np.array([[x, y]])

        try:
            alpha = self.svr_model_x.predict(features)[0]
            beta = self.svr_model_y.predict(features)[0]

            alpha = np.clip(alpha, padding, self.screen_width - padding)
            beta = np.clip(beta, padding, self.screen_height - padding)

            return np.array([alpha, beta])
        except Exception as e:
            print("SVR prediction error:", e)
            return None

    def calculate_consts_linear(self):

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
            "screen_position": []
        }
        l_iris_data = []
        was_collecting = False  # Tracks the previous state of the flag

        while not self.calibration.exit_event.is_set():
            if self.calibration.iris_data_flag:
                was_collecting = True
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")

                l_iris_data.append(l_iris_cent)

            else:
                # Only append once when data_flag switches from True to False
                if was_collecting:
                    l_median = np.median(l_iris_data, axis=0).astype(int).tolist()
                    iris_data_dict["l_iris_center"].append(l_median)

                    l_iris_data = []
                    was_collecting = False  # Reset flag to prevent repeated appending
            time.sleep(0.01)

        iris_data_dict["screen_position"] = self.screen_positions[1:]

        iris_data_list = []
        for l_iris, screen_pos in zip(iris_data_dict["l_iris_center"],
                                      self.screen_positions[1:]):
            iris_data_list.append({"l_iris_center": l_iris,
                                   "screen_position": screen_pos})

        return iris_data_list

    def validation_iris_data(self):

        self.calibration_data = self.read_from_file()
        self.calculate_consts_linear()
        self.train_polynomial_regression()
        self.train_svr()

        predictions = []
        was_collecting = False  # Tracks the previous state of the flag
        current_position = 1

        while not self.validation.exit_event.is_set():
            if self.validation.iris_data_flag:
                was_collecting = True
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")

                self.gaze = self.linear_mapping(l_iris_cent)
                # self.gaze = self.polynomial_mapping(l_iris_cent)
                # self.gaze = self.svr_mapping(l_iris_cent)

                predictions.append(self.gaze)

            else:
                if was_collecting:
                    metrics = self.calculate_metrics_per_target(predictions=predictions,
                                                                target_point=self.screen_positions[current_position])

                    # print(metrics) ###########################################################
                    self.all_metrics.append(metrics)

                    current_position = current_position + 1
                    predictions = []
                    was_collecting = False  # Reset flag to prevent repeated appending

            time.sleep(0.01)

        self.calculate_metrics_summary()

    def calculate_metrics_summary(self):

        if not self.all_metrics:
            print("No metrics to summarize.")
            return

        # Get all metric keys (assumes all dicts have the same keys)
        metric_keys = self.all_metrics[0].keys()

        summary = {}

        for key in metric_keys:
            values = [m[key] for m in self.all_metrics]
            summary[key] = {
                'mean': round(np.mean(values), 3),
                'min': round(np.min(values), 3),
                'max': round(np.max(values), 3)
            }

        # Optional: print results
        print("\nValidation Summary:")
        for key, stats in summary.items():
            print(f"{key}: mean={stats['mean']}, min={stats['min']}, max={stats['max']}")

        # Optionally store or return it
        self.metrics_summary = summary

    def calculate_metrics_per_target(self, predictions, target_point, screen_dpi=96, distance_cm=60):
        """
        predictions: list of (x, y) gaze points (in pixels)
        target_point: (x, y) position of the shown calibration point (in pixels)
        screen_dpi: screen resolution in dots per inch (default 96)
        distance_cm: estimated distance from user's eyes to screen in cm (default 60 cm)
        """

        predictions = np.array(predictions)
        target_point = np.array(target_point)

        # --- Pixels ---
        prediction_mean = np.mean(predictions, axis=0)
        accuracy_px = np.linalg.norm(prediction_mean - target_point)
        precision_px = np.std(predictions, axis=0)
        precision_px = np.linalg.norm(precision_px)

        # --- Centimeters ---
        px_to_cm = 2.54 / screen_dpi
        accuracy_cm = accuracy_px * px_to_cm
        precision_cm = precision_px * px_to_cm

        # --- Visual angle (degrees) ---
        accuracy_deg = math.degrees(math.atan(accuracy_cm / distance_cm))
        precision_deg = math.degrees(math.atan(precision_cm / distance_cm))

        return {
            'accuracy_px': accuracy_px,
            'precision_px': precision_px,
            'accuracy_cm': accuracy_cm,
            'precision_cm': precision_cm,
            'accuracy_deg': accuracy_deg,
            'precision_deg': precision_deg
        }

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




