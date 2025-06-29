import sys
import os
from src.gaze_tracker import GazeTracker


if __name__ == '__main__':
    gaze_tracker = GazeTracker()

    # Camera and detection threads
    processing_threads = [
        gaze_tracker.cam.get_feed_thread,
        gaze_tracker.cam.display_feed_thread,
        gaze_tracker.detector.face_mesh_thread,
    ]

    # Start camera and detector threads
    for thread in processing_threads:
        thread.start()

    # Run calibration (blocking until complete)
    gaze_tracker.calibration.calibration_gui_thread.start()
    gaze_tracker.calibration_data_thread.start()
    gaze_tracker.calibration.calibration_gui_thread.join()
    gaze_tracker.calibration_data_thread.join()

    # Run validation (optional, blocking)
    gaze_tracker.validation.validation_gui_thread.start()
    gaze_tracker.validation_data_thread.start()
    gaze_tracker.validation.draw_gaze_thread.start()
    gaze_tracker.validation.validation_gui_thread.join()
    gaze_tracker.validation_data_thread.join()
    gaze_tracker.validation.draw_gaze_thread.join()

    # Wait for camera and detector threads to finish
    for thread in processing_threads:
        thread.join()




