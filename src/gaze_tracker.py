import os
import sys
import threading
from camera import Camera
from detector import Detector
from sklearn.linear_model import LinearRegression


class GazeTracker:
    def __init__(self):
        self.env_cleanup()
        self.cam = Camera()
        self.detector = Detector(camera=self.cam)

        from gui.calibration import Calibration
        from gui.validation import Validation
        self.calibration = Calibration(self)
        self.validation = Validation(self)

        self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.exit_event = threading.Event()

    def linear_mapping(self, coordinates, ref_points):
        x, y = coordinates
        x1, y1, x2, y2, alpha1, beta1, alpha2, beta2 = ref_points

        # Linearno mapiranje za α (horizontalno)
        alpha = alpha1 + (x - x1) / (x2 - x1) * (alpha2 - alpha1)

        # Linearno mapiranje za β (vertikalno)
        beta = beta1 + (y - y1) / (y2 - y1) * (beta2 - beta1)

        return alpha, beta

    def predict_gaze(self, l_iris, r_iris, ref_points):
        alpha_l, beta_l = self.linear_mapping(l_iris, ref_points)
        alpha_r, beta_r = self.linear_mapping(r_iris, ref_points)

        # Prosek između levog i desnog oka
        alpha = (alpha_l + alpha_r) / 2
        beta = (beta_l + beta_r) / 2

        return alpha, beta

    def env_cleanup(self):
        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")




