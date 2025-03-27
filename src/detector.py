import sys
import cv2
import mediapipe as mp
import threading


class FaceRecognizer:
    def __init__(self, camera):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.8)

        self.camera = camera
        self.face_box_location = None

        self.detect_face_thread = threading.Thread(target=self.detect_face, daemon=True)

    def detect_face(self):
        """Continuously detects faces in the camera feed."""
        while True:
            if self.camera.feed is not None:
                # Convert BGR to RGB for MediaPipe processing
                feed_rgb = cv2.cvtColor(self.camera.feed, cv2.COLOR_BGR2RGB)

                # Process the image and detect faces
                results = self.face_detector.process(feed_rgb)

                if results.detections:
                    print("Face detected!")
                    for detection in results.detections:
                        # Get bounding box information
                        self.face_box_location = detection.location_data.relative_bounding_box
                        self.camera.face_box = self.face_box_location
                        #print(self.face_box_location)
                else:
                    print("No face detected.")
            else:
                self.stop()

    def stop(self):
        if self.detect_face_thread.is_alive():
            self.detect_face_thread.join()

        print("Detector successfully closed.")
