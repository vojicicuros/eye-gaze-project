import sys
import cv2
import mediapipe as mp
import threading


no_face_counter = 1

class FaceRecognizer:
    def __init__(self, camera):
        # FACE DETECTION MODEL
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.8)
        self.face_box_location = None

        # FACE MESH DETECTION MODEL
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]
        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]
        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]
        self.mesh_landmarks = {"left_eye": [], "right_eye": [], "left_iris": [], "right_iris": []}

        self.camera = camera

        self.detect_face_thread = threading.Thread(target=self.detect_face, daemon=True)
        self.face_mesh_thread = threading.Thread(target=self.detect_face_mesh, daemon=True)

    def detect_face(self):
        """Continuously detects face in the camera feed."""
        while True:
            if self.camera.running:
                # Convert BGR to RGB for MediaPipe processing
                feed_rgb = cv2.cvtColor(self.camera.feed, cv2.COLOR_BGR2RGB)

                # Process the image and detect faces
                results = self.face_detector.process(feed_rgb)

                if results.detections:
                    #print("Face detected!")
                    for detection in results.detections:
                        # Get bounding box information
                        self.face_box_location = detection.location_data.relative_bounding_box
                        self.camera.face_box = self.face_box_location
                else:
                    print("No face detected.")
            else:
                self.stop()

    def detect_face_mesh(self):
        while True:
            if self.camera.running:
                img_rgb = cv2.cvtColor(self.camera.feed, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(img_rgb)

                if results.multi_face_landmarks:
                    for face_lms in results.multi_face_landmarks:
                        for i, lm in enumerate(face_lms.landmark):
                            h, w, _ = self.camera.feed.shape
                            x, y = int(lm.x * w), int(lm.y * h)

                            if i in self.LEFT_EYE_LANDMARKS:
                                self.mesh_landmarks["left_eye"].append((x, y))
                            if i in self.RIGHT_EYE_LANDMARKS:
                                self.mesh_landmarks["right_eye"].append((x, y))
                            if i in self.LEFT_IRIS_LANDMARKS:
                                self.mesh_landmarks["left_iris"].append((x, y))
                            if i in self.RIGHT_IRIS_LANDMARKS:
                                self.mesh_landmarks["right_iris"].append((x, y))
                self.camera.face_landmarks = self.mesh_landmarks
                # Reset mesh_landmarks
                self.mesh_landmarks = {"left_eye": [], "right_eye": [], "left_iris": [], "right_iris": []}


            else:
                self.stop()
    def stop(self):
        if self.detect_face_thread.is_alive():
            self.detect_face_thread.join()
        if self.face_mesh_thread.is_alive():
            self.face_mesh_thread.join()
        print("Detector successfully closed.")

