import os
import sys
import threading

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

        #self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.exit_event = threading.Event()

    def predict(self, iris_landmarks):
        """
        Predict the screen position based on input iris landmarks.
        """
        prediction = self.model.predict([iris_landmarks])
        return prediction[0]

    def prepare_data(self, file_path):
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

    def validate(self, json_data):
        """
        Validate the model by calculating the error metrics on the calibration data.
        """
        features, actual_screen_positions = self.prepare_data(json_data)

        # Predict screen positions for the calibration data
        predicted_screen_positions = self.model.predict(features)

        # Calculate error metrics
        mae = mean_absolute_error(actual_screen_positions, predicted_screen_positions)
        mse = mean_squared_error(actual_screen_positions, predicted_screen_positions)
        rmse = np.sqrt(mse)

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        # Plot predicted vs actual screen positions for visualization
        actual_x = actual_screen_positions[:, 0]
        actual_y = actual_screen_positions[:, 1]
        predicted_x = predicted_screen_positions[:, 0]
        predicted_y = predicted_screen_positions[:, 1]

        # Plot x positions
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(actual_x, predicted_x, color='blue', label="Predicted vs Actual")
        plt.plot([min(actual_x), max(actual_x)], [min(actual_x), max(actual_x)], color='red', linestyle='--')
        plt.xlabel("Actual X")
        plt.ylabel("Predicted X")
        plt.title("Prediction for X Positions")
        plt.legend()

        # Plot y positions
        plt.subplot(1, 2, 2)
        plt.scatter(actual_y, predicted_y, color='green', label="Predicted vs Actual")
        plt.plot([min(actual_y), max(actual_y)], [min(actual_y), max(actual_y)], color='red', linestyle='--')
        plt.xlabel("Actual Y")
        plt.ylabel("Predicted Y")
        plt.title("Prediction for Y Positions")
        plt.legend()

        plt.tight_layout()
        plt.show()

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
        X, y = self.prepare_data(file_path=file_path)

        # Train model
        print("Training linear model.")
        self.model.fit(X, y)

        # Save model
        model_path = os.path.join("data", "linear_model.joblib")
        joblib.dump(self.model, model_path)
        print(f"Linear model trained and saved to {model_path}.")

    def save_iris_data(self, data):
        file_path = os.path.join("data", "iris_data.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            print("Saved data into .json file.")

    def env_cleanup(self):
        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")


    def calculate_positions(self, num_of_dots):
        monitor = get_monitors()[0]
        screen_width = monitor.width
        screen_height = monitor.height

        n = num_of_dots

        row_step = (screen_height - 2 * padding) // (n-1)
        col_step = (screen_width - 2 * padding) // (n-1)

        positions = [(screen_width // 2, screen_height // 2)] + [
            (padding + i * col_step, padding + j * row_step)
            for j in range(n) for i in range(n)
        ]
        return positions




