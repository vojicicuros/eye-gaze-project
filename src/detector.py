import cv2
import mediapipe as mp
from camera import Camera
import threading


class FaceRecognizer:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    def detect_face(self, camera):
        """Continuously detects faces in the camera feed."""
        while True:
            if camera.feed is not None:
                # Convert BGR to RGB for MediaPipe processing
                feed_rgb = cv2.cvtColor(camera.feed, cv2.COLOR_BGR2RGB)

                # Process the image and detect faces
                results = self.face_detector.process(feed_rgb)

                if results.detections:
                    print("Face detected!")
                else:
                    print("No face detected.")


if __name__ == '__main__':
    cam = Camera()
    recognizer = FaceRecognizer()

    t1 = threading.Thread(target=cam.get_feed, args=())
    t1.setDaemon(True)
    t1.start()

    t2 = threading.Thread(target=recognizer.detect_face, args=(cam,))
    t2.setDaemon(True)
    t2.start()

    t1.join()
    t2.join()