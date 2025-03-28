import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.camera import Camera
from src.detector import FaceRecognizer


if __name__ == '__main__':
    cam = Camera()
    recognizer = FaceRecognizer(camera=cam)

    cam.get_feed_thread.start()
    #recognizer.detect_face_thread.start()
    recognizer.face_mesh_thread.start()
    cam.display_feed_thread.start()

    cam.get_feed_thread.join()
    cam.display_feed_thread.join()
    #recognizer.detect_face_thread.join()
    recognizer.face_mesh_thread.join()

