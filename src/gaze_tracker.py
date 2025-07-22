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

        #self.method_num = 0  # 0 - linear mapping
                             # 1 - polynomial regression mapping
                             # 2 - SVR method mapping

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

    def train_polynomial_regression(self):
        """
        Train polynomial regression using iris center and eye corners as input features.
        """
        X = []
        Yx = []
        Yy = []

        for entry in self.calibration_data:
            ix, iy = entry["l_iris_center"]
            lx, ly = entry["l_eye_corner"]
            rx, ry = entry["r_eye_corner"]
            sx, sy = entry["screen_position"]

            # Optional: You can expand this with polynomial terms of all inputs if needed
            X.append([1, ix, iy, lx, ly, rx, ry])
            # X.append([
            #     1,
            #     ix, iy, lx, ly, rx, ry,
            #     ix ** 2, iy ** 2, lx ** 2, ly ** 2, rx ** 2, ry ** 2,
            #     ix * iy, ix * lx, ix * ly, ix * rx, ix * ry,
            #     iy * lx, iy * ly, iy * rx, iy * ry,
            #     lx * ly, lx * rx, lx * ry,
            #     ly * rx, ly * ry,
            #     rx * ry
            # ])

            Yx.append(sx)
            Yy.append(sy)

        X = np.array(X)
        Yx = np.array(Yx)
        Yy = np.array(Yy)

        self.poly_reg_x = LinearRegression().fit(X, Yx)
        self.poly_reg_y = LinearRegression().fit(X, Yy)

        print("Polynomial regression trained.")

    def train_svr(self):
        """
        Train SVR model using iris center and eye corners as features.
        """
        X = []
        Yx = []
        Yy = []

        for entry in self.calibration_data:
            ix, iy = entry["l_iris_center"]
            lx, ly = entry["l_eye_corner"]
            rx, ry = entry["r_eye_corner"]
            sx, sy = entry["screen_position"]

            X.append([ix, iy, lx, ly, rx, ry])
            Yx.append(sx)
            Yy.append(sy)

        X = np.array(X)
        Yx = np.array(Yx)
        Yy = np.array(Yy)

        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR

        self.svr_model_x = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1000, epsilon=0.5))
        self.svr_model_y = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1000, epsilon=0.5))

        self.svr_model_x.fit(X, Yx)
        self.svr_model_y.fit(X, Yy)

        print("SVR model trained.")

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

    def polynomial_mapping(self, iris_center, l_eye_corner, r_eye_corner):
        """
        Predict the gaze point using polynomial regression with extra eye corner features.
        """
        if iris_center is None or l_eye_corner is None or r_eye_corner is None:
            return None

        ix, iy = iris_center
        lx, ly = l_eye_corner
        rx, ry = r_eye_corner

        features = np.array([[1, ix, iy, lx, ly, rx, ry]])

        try:
            alpha = self.poly_reg_x.predict(features)[0]
            beta = self.poly_reg_y.predict(features)[0]

            alpha = np.clip(alpha, padding, self.screen_width - padding)
            beta = np.clip(beta, padding, self.screen_height - padding)
            return np.array([alpha, beta])
        except Exception as e:
            print("Polynomial mapping error:", e)
            return None

    def svr_mapping(self, iris_center, left_corner, right_corner):
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
        if iris_center is None or left_corner is None or right_corner is None:
            return None

        ix, iy = iris_center
        lx, ly = left_corner
        rx, ry = right_corner
        features = np.array([[ix, iy, lx, ly, rx, ry]])

        try:
            alpha = self.svr_model_x.predict(features)[0]
            beta = self.svr_model_y.predict(features)[0]

            alpha = np.clip(alpha, padding, self.screen_width - padding)
            beta = np.clip(beta, padding, self.screen_height - padding)

            return np.array([alpha, beta])
        except Exception as e:
            print("SVR prediction error:", e)
            return None

    def calibration_iris_data(self):
        iris_data_dict = {
            "l_iris_center": [],
            "screen_position": [],
            "l_eye_corner": [],
            "r_eye_corner": []
        }
        l_iris_data, left_corner, right_corner = [], [], []
        was_collecting = False  # Tracks the previous state of the flag

        while not self.calibration.exit_event.is_set():
            if self.calibration.iris_data_flag:
                was_collecting = True
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")
                l_iris_data.append(l_iris_cent)

                eye_contour = self.detector.camera.eyes_landmarks.get("left_eye")
                left, right = self.detector.get_eye_corners(eye_input=eye_contour)
                left_corner.append(left)
                right_corner.append(right)
            else:
                # Only append once when data_flag switches from True to False
                if was_collecting:
                    l_median = np.median(l_iris_data, axis=0).astype(int).tolist()
                    l_corner_median = np.median(left_corner, axis=0).astype(int).tolist()
                    r_corner_median = np.median(right_corner, axis=0).astype(int).tolist()
                    iris_data_dict["l_iris_center"].append(l_median)
                    iris_data_dict["l_eye_corner"].append(l_corner_median)
                    iris_data_dict["r_eye_corner"].append(r_corner_median)

                    l_iris_data, left_corner, right_corner = [], [], []
                    was_collecting = False  # Reset flag to prevent repeated appending
            time.sleep(0.01)

        iris_data_dict["screen_position"] = self.screen_positions[1:]

        iris_data_list = []
        for l_iris, l_corner, r_corner, screen_pos in zip(
                iris_data_dict["l_iris_center"],
                iris_data_dict["l_eye_corner"],
                iris_data_dict["r_eye_corner"],
                iris_data_dict["screen_position"]):
            iris_data_list.append({
                "l_iris_center": l_iris,
                "l_eye_corner": l_corner,
                "r_eye_corner": r_corner,
                "screen_position": screen_pos
            })
        return iris_data_list

    def calibration_iris_data_wrap(self):

        iris_data = self.calibration_iris_data()
        self.save_data_to_file(data=iris_data)

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
                eye_center_input = self.detector.camera.eyes_landmarks.get("l_iris_center")
                eye_outer_input = self.detector.camera.eyes_landmarks.get("left_eye")
                l_corner_input, r_corner_input = self.detector.get_eye_corners(eye_outer_input)

                if method_num == 0:
                    self.gaze = self.linear_mapping(eye_center_input)
                elif method_num == 1:
                    self.gaze = self.polynomial_mapping(eye_center_input, l_corner_input, r_corner_input)
                elif method_num == 2:
                    self.gaze = self.svr_mapping(eye_center_input, l_corner_input, r_corner_input)

                predictions.append(self.gaze)

            else:
                if was_collecting:
                    metrics = self.calculate_metrics_per_target(predictions=predictions,
                                                                target_point=self.screen_positions[current_position])
                    self.all_metrics.append(metrics)

                    current_position = current_position + 1
                    predictions = []
                    was_collecting = False  # Reset flag to prevent repeated appending

            time.sleep(0.01)

        self.calculate_metrics_summary()

    def calculate_metrics_per_target(self, predictions, target_point, screen_dpi=81, distance_cm=60):
        """
        predictions: list of (x, y) gaze points (in pixels)
        target_point: (x, y) position of the shown calibration point (in pixels)
        screen_dpi: screen resolution in dots per inch (default 81 for 27" monitor
                                                        default 91 for 24" monitor)
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

    def calculate_metrics_summary(self):
        if not self.all_metrics:
            print("No metrics to summarize.")
            return

        # Get all metric keys (assumes all dicts have the same keys)
        metric_keys = self.all_metrics[0].keys()

        # Map each metric to its unit
        unit_map = {
            'accuracy_px': 'px',
            'precision_px': 'px',
            'accuracy_cm': 'cm',
            'precision_cm': 'cm',
            'accuracy_deg': '°',
            'precision_deg': '°'
        }

        summary = {}

        for key in metric_keys:
            values = [m[key] for m in self.all_metrics]
            summary[key] = {
                'mean': round(np.mean(values), 3),
                'min': round(np.min(values), 3),
                'max': round(np.max(values), 3),
                'unit': unit_map.get(key, '')
            }

        # Print results with units
        print("\nValidation Summary:")
        for key, stats in summary.items():
            u = stats['unit']
            print(f"{key} [{u}]: mean={stats['mean']}{u}, min={stats['min']}{u}, max={stats['max']}{u}")

        # Store for later if needed
        self.metrics_summary = summary

    def calculate_positions(self, n=num_of_dots):

        row_step = (screen_height - 2 * padding) // (n-1)
        col_step = (screen_width - 2 * padding) // (n-1)

        positions = [(screen_width // 2, screen_height // 2)] + [
            (padding + i * col_step, padding + j * row_step)
            for j in range(n) for i in range(n)
        ]
        return positions

    def save_data_to_file(self, data):
        try:
            os.makedirs("data", exist_ok=True)  # Ensure directory exists

            file_path = os.path.join("data", filename)

            # Read existing data
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                        if not isinstance(existing_data, list):
                            print(f"Warning: Overwriting non-list data in {filename}")
                            existing_data = []
                    except json.JSONDecodeError:
                        print(f"⚠Warning: Corrupted JSON in {filename}. Overwriting file.")
                        existing_data = []
            else:
                existing_data = []

            # Append and save
            existing_data.extend(data)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)

            print(f"Saved data into {file_path}")

        except Exception as e:
            print(f"Error saving file {filename}: {e}")

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

    def env_cleanup(self):
        # json file must be in data folder

        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")





