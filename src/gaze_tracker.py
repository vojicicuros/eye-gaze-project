import os
import sys
import threading
from camera_feed import Camera
from landmark_detector import Detector
from sklearn.linear_model import LinearRegression
import numpy as np
import json

class GazeTracker:
    def __init__(self):
        self.env_cleanup()
        self.cam = Camera()
        self.detector = Detector(camera=self.cam)

        # Initialize the model
        self.model = LinearRegression()

        from gui.calibration import Calibration
        from gui.validation import Validation

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

    def train_linear_model(self):
        file_path = os.path.join("data", "iris_data.json")
        X, y = self.prepare_data(file_path=file_path)

        # Train model
        print("Training linear model.")
        self.model.fit(X, y)
        print("Linear model trained.")

    def env_cleanup(self):
        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")




