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

    # Run calibration part (blocking until complete)
    gaze_tracker.calibration.calibration_gui_thread.start()
    gaze_tracker.calibration_data_thread.start()

    gaze_tracker.calibration.calibration_gui_thread.join()
    gaze_tracker.calibration_data_thread.join()

    # Run validation part (optional, blocking)
    gaze_tracker.validation.validation_gui_thread.start()
    gaze_tracker.validation_data_thread.start()
    gaze_tracker.validation.draw_gaze_gui_thread.start()

    gaze_tracker.validation.validation_gui_thread.join()
    gaze_tracker.validation_data_thread.join()
    gaze_tracker.validation.draw_gaze_gui_thread.join()

    # Run Gazing part
    gaze_tracker.gazing_part.gazing_data_thread.start()
    gaze_tracker.gazing_part.gazing_gui_thread.start()
    gaze_tracker.gazing_part.draw_gaze_thread.start()

    gaze_tracker.gazing_part.gazing_gui_thread.join()

    # Wait for camera and detector threads to finish
    for thread in processing_threads:
        thread.join()

    print("All done!")

