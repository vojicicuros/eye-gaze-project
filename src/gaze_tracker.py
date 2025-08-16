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
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline


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

        self.poly_reg_y = None
        self.poly_reg_x = None

        self.x1 = self.x2 = self.alpha1 = self.alpha2 = 0
        self.y1 = self.y2 = self.beta1 = self.beta2 = 0

        self.calibration_data = None
        self.gaze = None

        self.all_metrics = []
        self.metrics_summary = None
        self.metrics_summary_table = None

        self.screen_positions = self.calculate_positions(num_of_dots)
        self.calibration = Calibration(self)
        self.validation = Validation(self)
        self.gazing_part = Gazing(self)

        # self.predict_gaze_thread = threading.Thread(target=self.predict_gaze)
        self.calibration_data_thread = threading.Thread(target=self.calibration_iris_data_wrap, daemon=True)
        self.validation_data_thread = threading.Thread(target=self.validation_iris_data, daemon=True)
        self.exit_event = threading.Event()

    def calculate_consts_linear_map(self):

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

    def linear_mapping(self, live_data):

        x, y = live_data

        # Clip withing [x1,x2] and [y1,y2]
        x = np.clip(x, self.x1, self.x2)
        y = np.clip(y, self.y1, self.y2)

        # Linear interpolation
        alpha = self.alpha1 + (x - self.x1) / (self.x2 - self.x1) * (self.alpha2 - self.alpha1)
        beta = self.beta1 + (y - self.y1) / (self.y2 - self.y1) * (self.beta2 - self.beta1)

        # Clip with padding
        alpha = np.clip(alpha, padding, self.screen_width - padding)
        alpha = np.abs(self.screen_width - padding - alpha)
        beta = np.clip(beta, padding, self.screen_height - padding)

        return np.array([alpha, beta])

    def train_polynomial_regression(self, poly_degree=2):
        """
        Train robust polynomial regression for gaze estimation.
        Based on method from 'A robust method for calibration of eye tracking data' (the PDF).
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

        # Robust polynomial regression (degree=2)
        self.poly_reg_x = Pipeline([
            ("scaler", RobustScaler()),
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))
        ])
        self.poly_reg_y = Pipeline([
            ("scaler", RobustScaler()),
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5))
        ])

        self.poly_reg_x.fit(X, Yx)
        self.poly_reg_y.fit(X, Yy)

        print("Robust polynomial regression trained (degree={})".format(poly_degree))

    def polynomial_mapping(self, iris_center, l_eye_corner, r_eye_corner):
        """
        Predict gaze point using robust polynomial regression.
        """
        if iris_center is None or l_eye_corner is None or r_eye_corner is None:
            return None

        ix, iy = iris_center
        lx, ly = l_eye_corner
        rx, ry = r_eye_corner

        features = np.array([[ix, iy, lx, ly, rx, ry]])

        try:
            alpha = self.poly_reg_x.predict(features)[0]
            beta = self.poly_reg_y.predict(features)[0]

            alpha = np.clip(alpha, padding, self.screen_width - padding)
            beta = np.clip(beta, padding, self.screen_height - padding)
            return np.array([alpha, beta])
        except Exception as e:
            print("Polynomial mapping error:", e)
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
        self.calculate_consts_linear_map()
        self.train_polynomial_regression()

        # scaler = self.poly_reg_x.named_steps["scaler"]
        # self.plot_scaling_grid(self.calibration_data, fitted_scaler=scaler, title_prefix="poly/")

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

        label = 'Linearno mapiranje' if method_num == 0 else (
            'Polinomijalna regresija' if method_num == 1 else 'SVR')

        self.calculate_metrics_summary(
            save_csv_path='results/metrics_summary.csv',
            method_label=label
        )

        self.calculate_stability_summary(
            save_csv_path='results/metrics_stability.csv',
            method_label=label,
            include_magnitude=True
        )

    def calculate_metrics_per_target(self, predictions, target_point, screen_dpi=81, distance_cm=60):
        predictions = np.array(predictions)
        target_point = np.array(target_point)

        # --- Pixels ---
        prediction_mean = np.mean(predictions, axis=0)
        accuracy_px = np.linalg.norm(prediction_mean - target_point)
        precision_px = np.linalg.norm(np.std(predictions, axis=0))

        # std po osama
        std_x = np.std(predictions[:, 0])
        std_y = np.std(predictions[:, 1])

        # bias po osama
        bias_x = prediction_mean[0] - target_point[0]
        bias_y = prediction_mean[1] - target_point[1]

        # --- Centimeters ---
        px_to_cm = 2.54 / screen_dpi
        accuracy_cm = accuracy_px * px_to_cm
        precision_cm = precision_px * px_to_cm

        # --- Visual angle (degrees) ---
        accuracy_deg = math.degrees(math.atan(accuracy_cm / distance_cm))
        precision_deg = math.degrees(math.atan(precision_cm / distance_cm))

        # --- p95 ---
        errors_px = np.linalg.norm(predictions - target_point, axis=1)
        errors_cm = errors_px * px_to_cm
        errors_deg = np.degrees(np.arctan(errors_cm / distance_cm))

        p95_px = np.percentile(errors_px, 95)
        p95_deg = np.percentile(errors_deg, 95)

        return {
            'accuracy_px': accuracy_px,
            'accuracy_cm': accuracy_cm,
            'accuracy_deg': accuracy_deg,
            'precision_px': precision_px,
            'precision_cm': precision_cm,
            'precision_deg': precision_deg,
            'std_x': std_x,
            'std_y': std_y,
            'bias_x': bias_x,
            'bias_y': bias_y,
            'p95_px': p95_px,
            'p95_deg': p95_deg
        }

    def _metrics_table(self, metrics_list):
        if not metrics_list:
            return None
        import pandas as pd

        df = pd.DataFrame(metrics_list)

        # SVI ključevi koje očekujemo u metrics_list dict-ovima:
        cols_order = {
            'accuracy_px': 'Tačnost [px]',
            'accuracy_cm': 'Tačnost [cm]',
            'accuracy_deg': 'Tačnost [°]',
            'precision_px': 'Preciznost [px]',
            'precision_cm': 'Preciznost [cm]',
            'precision_deg': 'Preciznost [°]',
        }

        # --- DIJAGNOSTIKA (pomaže kad "neće") ---
        # Proveri da li neki ključ fali u df:
        missing = [k for k in cols_order if k not in df.columns]
        if missing:
            raise KeyError(
                f"Nedostaju sledeći ključevi u all_metrics: {missing}. "
                f"Dobijeni ključevi: {list(df.columns)}"
            )

        use_cols = list(cols_order.keys())

        mean_row = df[use_cols].mean()
        min_row = df[use_cols].min()
        max_row = df[use_cols].max()

        out = (
            pd.DataFrame([mean_row, min_row, max_row],
                         index=['Srednja vrednost', 'Minimum', 'Maksimum'])
            .rename(columns=cols_order)
            .applymap(lambda x: round(float(x), 3))
        )
        return out

    def _metrics_table_stability(self, metrics_list, include_magnitude=True):
        """
        Pravi zasebnu tabelu za stabilnost/sistematske metrike:
          Std X/Y [px], Bias X/Y [px], p95 [px], p95 [°]
        Opciono dodaje magnitude (RMS std i |bias|).
        """
        if not metrics_list:
            return None
        import pandas as pd
        import numpy as np

        df = pd.DataFrame(metrics_list)

        # Osnovne kolone
        cols_order = {
            'std_x': 'Std X [px]',
            'std_y': 'Std Y [px]',
            'bias_x': 'Bias X [px]',
            'bias_y': 'Bias Y [px]',
            'p95_px': 'p95 [px]',
            'p95_deg': 'p95 [°]',
        }

        # (opciono) dodaj magnitude: RMS std i |bias|
        if include_magnitude:
            # Napravi kolone ako ne postoje
            if 'std_mag' not in df.columns:
                df['std_mag'] = np.sqrt(np.square(df.get('std_x', 0)) + np.square(df.get('std_y', 0)))
            if 'bias_mag' not in df.columns:
                df['bias_mag'] = np.sqrt(np.square(df.get('bias_x', 0)) + np.square(df.get('bias_y', 0)))

            cols_order.update({
                'std_mag': 'Std mag [px]',
                'bias_mag': 'Bias mag [px]',
            })

        # Provera ključeva
        missing = [k for k in cols_order if k not in df.columns]
        if missing:
            raise KeyError(f"Nedostaju ključevi za stability tabelu: {missing}")

        use_cols = list(cols_order.keys())
        mean_row = df[use_cols].mean()
        min_row = df[use_cols].min()
        max_row = df[use_cols].max()

        out = (
            pd.DataFrame([mean_row, min_row, max_row],
                         index=['Srednja vrednost', 'Minimum', 'Maksimum'])
            .rename(columns=cols_order)
            .applymap(lambda x: round(float(x), 3))
        )
        return out

    def calculate_stability_summary(self, save_csv_path=None, method_label=None, include_magnitude=True):
        """
        Ispiše i (opciono) snimi stability tabelu (std/bias/p95).
        """
        if not getattr(self, 'all_metrics', None):
            print("No metrics to summarize.")
            return None

        table = self._metrics_table_stability(self.all_metrics, include_magnitude=include_magnitude)
        title = f"Tabela – stabilnost i sistematske metrike{f' ({method_label})' if method_label else ''}"
        print("\n" + title)
        print(table.to_string())

        if save_csv_path:
            table.to_csv(save_csv_path, index=True, encoding='utf-8-sig')

        self.metrics_stability_table = table
        return table

    def calculate_metrics_summary(self, save_csv_path=None, method_label=None):
        """
        - Prikaže tabelu sa Srednja/Minimum/Maksimum po metriki.
        - Ako su prosleđene putanje, snimi CSV

        save_csv_path: npr. 'results/linear_summary.csv'
        method_label : Linearno mapiranje ili Polinomijalna regresija
        """
        if not getattr(self, 'all_metrics', None):
            print("No metrics to summarize.")
            return None

        table = self._metrics_table(self.all_metrics)
        if table is None:
            print("No metrics to summarize.")
            return None

        title = f"Tabela – statistika tačnosti i preciznosti{f' ({method_label})' if method_label else ''}"
        print("\n" + title)
        print(table.to_string())

        if save_csv_path:
            table.to_csv(save_csv_path, index=True, encoding='utf-8-sig')

        self.metrics_summary_table = table
        return table

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
                        print(f"Warning: Corrupted JSON in {filename}. Overwriting file.")
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
        # Json file must be in data folder

        file_path = os.path.join("data", "iris_data.json")
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            "No json file."
        print("Environment is fine.")

    def _extract_X_from_calibration(self, calibration_data):
        """
        calibration_data: lista dict-ova sa ključevima:
          - "l_iris_center": [ix, iy]
          - "l_eye_corner":  [lx, ly]
          - "r_eye_corner":  [rx, ry]
        """
        X = []
        for e in calibration_data:
            ix, iy = e["l_iris_center"]
            lx, ly = e["l_eye_corner"]
            rx, ry = e["r_eye_corner"]
            X.append([ix, iy, lx, ly, rx, ry])
        X = np.array(X, dtype=float)
        names = ["ix", "iy", "lx", "ly", "rx", "ry"]
        return X, names

    def plot_scaling_grid(self, calibration_data, fitted_scaler: RobustScaler = None, title_prefix=""):
        """
        Pravi 3x2 grid boxplotova sa *dve y-ose* po subplotu (pre/posle).
        Ako je fitted_scaler None: fituje novi RobustScaler na X i koristi ga.
        Ako je dat fitted_scaler: koristi baš taj (npr. iz pipeline-a).
        """
        X, names = self._extract_X_from_calibration(calibration_data)

        if fitted_scaler is None:
            scaler = RobustScaler()
            Xs = scaler.fit_transform(X)
            print("RobustScaler: fitovan na kalibracionim podacima.")
        else:
            scaler = fitted_scaler
            Xs = scaler.transform(X)
            print("RobustScaler: korišćen već istreniran scaler iz pipeline-a.")

        # Raspored po redovima
        layout = [["ix", "iy"],
                  ["lx", "ly"],
                  ["rx", "ry"]]

        fig, axes = plt.subplots(3, 2, figsize=(11, 9))
        axes = np.asarray(axes)

        # mapiranje imena na indeks kolone u X
        col_idx = {n: i for i, n in enumerate(names)}

        for r, row in enumerate(layout):
            for c, feat in enumerate(row):
                ax = axes[r, c]
                j = col_idx[feat]

                pre_vals = X[:, j]
                post_vals = Xs[:, j]

                ax2 = ax.twinx()  # druga y-osa za "posle"

                # Boxplot "pre" (levo)
                ax.boxplot(pre_vals, positions=[1], widths=0.4)
                ax.set_ylabel("pre scale")

                # Boxplot "posle" (desno)
                ax2.boxplot(post_vals, positions=[2], widths=0.4)
                ax2.set_ylabel("posle scale")

                # Podešavanje x ose
                ax.set_xticks([1, 2])
                ax.set_xticklabels(["pre", "posle"])

                # Opsezi y-osa nezavisni (svaka skala priča “svoj jezik”)
                # Ako želiš fiksni opseg, odkomentariši i prilagodi:
                # ax.set_ylim(np.min(pre_vals), np.max(pre_vals))
                # ax2.set_ylim(np.min(post_vals), np.max(post_vals))

                ax.set_title(f"{title_prefix}{feat}")

        plt.tight_layout()
        plt.show()

        # opciono: vrati scaler ako je fitovan ovde
        return scaler





