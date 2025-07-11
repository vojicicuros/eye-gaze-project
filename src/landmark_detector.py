import cv2
import mediapipe as mp
import threading
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh


# from camera_feed import Camera


class Smoother:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.smoothed = None

    def update(self, new_value):

        if self.smoothed is None:
            self.smoothed = np.array(new_value, dtype=np.float32)
        else:
            self.smoothed = self.alpha * np.array(new_value, dtype=np.float32) + (1 - self.alpha) * self.smoothed

        return self.smoothed.astype(int)


class Detector:
    def __init__(self, camera):
        # camera
        self.camera = camera

        self.eye_box_initialized = False
        self.eye_crop_width = 100
        self.eye_crop_height = 50

        # FACE DETECTION MODEL
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.8)

        # FACE MESH DETECTION MODEL
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False,
                                                    max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.7,
                                                    min_tracking_confidence=0.9)

        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]
        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]

        self.mesh_landmarks = {"left_eye": [],
                               "left_iris": [],
                               "l_iris_center": [],
                               }

        self.eye_smoothers = {
            "left_eye": Smoother(alpha=0.7),
            "left_iris": Smoother(alpha=0.7),
        }

        self.face_mesh_thread = threading.Thread(target=self.detect_mesh, daemon=True)

        print('Landmark detector setup successful.')

    def calc_iris_center(self, iris_landmarks):
        # Racunanje centra duzice pomocu preseka dve duzi

        x1, y1 = iris_landmarks[0]
        x2, y2 = iris_landmarks[1]
        x3, y3 = iris_landmarks[2]
        x4, y4 = iris_landmarks[3]

        k1 = (y3 - y1) / (x3 - x1) if x3 != x1 else float('inf')
        k2 = (y4 - y2) / (x4 - x2) if x4 != x2 else float('inf')

        if k1 == float('inf'):
            x_p = x1
            y_p = y2 + k2 * (x_p - x2)
        elif k2 == float('inf'):
            x_p = x2
            y_p = y1 + k1 * (x_p - x1)
        else:
            x_p = (k1 * x1 - k2 * x2 + y2 - y1) / (k1 - k2)
            y_p = y1 + k1 * (x_p - x1)
        x_p = int(x_p)
        y_p = int(y_p)
        return x_p, y_p

    def get_eye_corners(self, eye_input):
        """
        Returns the most left and most right points from the smoothed left eye landmarks.
        """
        # if not eye_input or len(eye_input) < 2:
        #     return None, None
        # Sort based on X coordinate
        sorted_eye = sorted(eye_input, key=lambda p: p[0])
        left_corner = sorted_eye[0]
        right_corner = sorted_eye[-1]
        return left_corner, right_corner

    def detect_mesh(self):

        # while self.camera.face_box is None:
        #     self.camera.face_box = self.detect_face_box()

        while True:
            if self.camera.running:
                img_rgb = cv2.cvtColor(self.camera.feed, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(img_rgb)
                new_landmarks = {"left_eye": [], "left_iris": []}

                if results.multi_face_landmarks:
                    face_lms = results.multi_face_landmarks[0]
                    new_landmarks = {"left_eye": [], "left_iris": []}

                    h, w, _ = self.camera.feed.shape
                    for i, lm in enumerate(face_lms.landmark):
                        x, y = int(lm.x * w), int(lm.y * h)

                        if i in self.LEFT_EYE_LANDMARKS:
                            new_landmarks["left_eye"].append((x, y))
                        if i in self.LEFT_IRIS_LANDMARKS:
                            new_landmarks["left_iris"].append((x, y))

                    # Apply smoothing
                    for key in self.mesh_landmarks.keys():
                        if key == "l_iris_center":
                            continue
                        if new_landmarks[key]:
                            self.mesh_landmarks[key] = self.eye_smoothers[key].update(new_landmarks[key])

                self.mesh_landmarks["l_iris_center"] = self.calc_iris_center(new_landmarks["left_iris"])
                self.camera.eyes_landmarks = self.mesh_landmarks.copy()

                # Reset temp mesh_landmarks
                self.mesh_landmarks = {"left_eye": [], "left_iris": [], "l_iris_center": []}

            else:
                self.stop()

    def stop(self):
        if self.face_mesh_thread.is_alive():
            self.face_mesh_thread.join()
        print("Detector successfully closed.")

