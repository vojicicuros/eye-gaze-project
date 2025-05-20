import os
import threading
import time
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
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


def prepare_data(file_path):
    """
    Prepare the data from the json. Extract iris landmarks and screen positions.
    """
    print(f"Extracting data from {file_path} file.")
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    features = []
    screen_positions = []

    for entry in json_data:
        # Extract iris landmarks
        l_iris_center = entry["l_iris_center"]
        r_iris_center = entry["r_iris_center"]

        # Combine iris boundary points and centers (10 points total)
        feature_vector = l_iris_center + r_iris_center

        # Extract screen position
        screen_position = entry["screen_position"]

        # Append to feature list
        features.append(feature_vector)
        screen_positions.append(screen_position)

    # Convert to numpy arrays for model training
    features = np.array(features)
    screen_positions = np.array(screen_positions)

    print("Data extracted.")
    return features, screen_positions


class GazeTracker:
    def __init__(self):
        self.env_cleanup()
        self.cam = Camera()
        self.detector = Detector(camera=self.cam)

        # Initialize the model
        self.model = self.import_model()

        from gui.calibration import Calibration
        from gui.validation import Validation

        self.screen_positions = self.calculate_positions(num_of_dots)
        self.calibration = Calibration(self)
        self.validation = Validation(self)

        # self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.calibration_data_thread = threading.Thread(target=self.calibration_iris_data_to_file)
        self.validation_data_thread = threading.Thread(target=self.validation_iris_data)
        self.exit_event = threading.Event()

    def linear_estimation(self, live_data, calibration_data):
        """
        """

        x, y = live_data[0], live_data[1]

        x1 = calibration_data[3]['iris_center'][0]
        x2 = calibration_data[5]['iris_center'][0]
        alpha1 = calibration_data[3]['screen_position'][0]
        alpha2 = calibration_data[5]['screen_position'][0]

        y1 = calibration_data[1]['iris_center'][1]
        y2 = calibration_data[7]['iris_center'][1]
        beta1 = calibration_data[1]['screen_position'][1]
        beta2 = calibration_data[7]['screen_position'][1]

        alpha = alpha1 + (x - x1) / (x2 - x1) * (alpha2 - alpha1)
        beta = beta1 + (y - y1) / (y2 - y1) * (beta2 - beta1)

        return np.array([alpha, beta])


    def import_model(self):
        model_path = os.path.join("data", "linear_model.joblib")
        if os.path.exists(model_path):
            print(f"Loading pretrained model from {model_path}.")
            return joblib.load(model_path)
        else:
            print("No pretrained model found. Using a new one.")
            return LinearRegression()

    def train_linear_model(self):
        file_path = os.path.join("data", "iris_data.json")
        X, y = prepare_data(file_path=file_path)

        # Train model
        print("Training linear model.")
        self.model.fit(X, y)

        # Save model
        model_path = os.path.join("data", "linear_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"Linear model trained and saved to {model_path}.")

    def get_iris_data(self, exit_event, data_flag, calibration_flag):

        iris_data_dict = {
            "l_iris_center": [],
            "r_iris_center": [],
            "screen_position": []
        }
        l_iris_data = []
        r_iris_data = []
        was_collecting = False  # Tracks the previous state of the flag

        while not exit_event.is_set():
            if self.validation.iris_data_flag:
                was_collecting = True
                l_iris_cent = self.detector.camera.eyes_landmarks.get("l_iris_center")
                r_iris_cent = self.detector.camera.eyes_landmarks.get("r_iris_center")

                avg_center = np.average([l_iris_cent, r_iris_cent], axis=0)
                print(avg_center)

                if l_iris_cent is not None:
                    l_iris_data.append(l_iris_cent)
                if r_iris_cent is not None:
                    r_iris_data.append(r_iris_cent)
            else:
                # Only append once when data_flag switches from True to False
                if was_collecting:
                    if l_iris_data and r_iris_data:
                        iris_data_dict["l_iris_center"].append(np.median(l_iris_data, axis=0).astype(int).tolist())
                        iris_data_dict["r_iris_center"].append(np.median(r_iris_data, axis=0).astype(int).tolist())
                    l_iris_data = []
                    r_iris_data = []
                    was_collecting = False  # Reset flag to prevent repeated appending
            time.sleep(0.01)

        # If calibration is running, set regular screen position; If validation is running, predict screen position
        if calibration_flag:
            iris_data_dict["screen_position"] = self.screen_positions[1:]
        else:
            input_data = iris_data_dict["l_iris_center"] + iris_data_dict["r_iris_center"]
            predicted_data = self.predict(input_data)
            iris_data_dict["screen_position"] = predicted_data

        iris_data_list = []
        for l_iris, r_iris, screen_pos in zip(iris_data_dict["l_iris_center"],
                                              iris_data_dict["r_iris_center"],
                                              self.screen_positions[1:]):
            iris_data_list.append({"l_iris_center": l_iris, "r_iris_center": r_iris, "screen_position": screen_pos})

        return iris_data_list

    def validation_iris_data(self):

        iris_data = self.get_iris_data(exit_event=self.validation.exit_event,
                                       data_flag=self.validation.iris_data_flag,
                                       calibration_flag=False)

        self.save_data_to_file(data=iris_data, filename="iris_data_val.json")

    def calibration_iris_data_to_file(self):

        iris_data = self.get_iris_data(exit_event=self.calibration.exit_event,
                                       data_flag=self.calibration.iris_data_flag,
                                       calibration_flag=True)

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




