import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.camera_feed import Camera
from src.landmark_detector import Detector
from src.gaze_tracker import GazeTracker
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))


if __name__ == '__main__':
    gaze_tracker = GazeTracker()


    # Start all threads
    threads = [
        gaze_tracker.calibration.start_calibration_thread,
        gaze_tracker.calibration.iris_data_thread,
        gaze_tracker.cam.get_feed_thread,
        gaze_tracker.cam.display_feed_thread,
        gaze_tracker.detector.detect_face_thread,
        gaze_tracker.detector.face_mesh_thread,
        # gaze_tracker.validation.start_validation_thread,
    ]

    print("Starting all threads...")
    for thread in threads:
        thread.start()

    print("Waiting for all threads to complete...")
    for thread in threads:
        thread.join()



