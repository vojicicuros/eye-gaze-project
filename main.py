import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.camera_feed import Camera
from src.landmark_detector import Detector
from src.gaze_tracker import GazeTracker
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))


if __name__ == '__main__':
    gaze_tracker = GazeTracker()
    gaze_tracker.calibration.start_calibration_thread.start()
    gaze_tracker.calibration.iris_data_thread.start()

    gaze_tracker.cam.get_feed_thread.start()
    #gaze_tracker.cam.display_feed_thread.start()
    gaze_tracker.detector.detect_face_thread.start()
    gaze_tracker.detector.face_mesh_thread.start()

    gaze_tracker.cam.get_feed_thread.join()
    #gaze_tracker.cam.display_feed_thread.join()
    gaze_tracker.detector.detect_face_thread.join()
    gaze_tracker.detector.face_mesh_thread.join()


