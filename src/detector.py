import sys
import cv2
import mediapipe as mp
import threading


class FaceRecognizer:
    def __init__(self, camera):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.8)

        # Initialize the parameters for face mesh detection
        self.static_image_mode = False  # Whether to process images (True) or video stream (False)
        self.max_num_faces = 1  # Maximum number of faces to detect
        self.refine_landmarks = False  # Whether to refine iris landmarks for better precision
        self.min_detection_con = 0.5  # Minimum confidence for face detection
        self.min_tracking_con = 0.5

        # Initialize Mediapipe FaceMesh solution
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.static_image_mode,
                                                 self.max_num_faces,
                                                 self.refine_landmarks,
                                                 self.min_detection_con,
                                                 self.min_tracking_con)

        # Store the landmark indices for specific facial features
        # These are predefined Mediapipe indices for left and right eyes, iris, nose, and mouth

        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]  # Left eye landmarks

        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]  # Right eye landmarks

        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]  # Left iris landmarks
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]  # Right iris landmarks

        self.NOSE_LANDMARKS = [193, 168, 417, 122, 351, 196, 419, 3, 248, 236, 456, 198, 420, 131, 360, 49, 279, 48,
                               278, 219, 439, 59, 289, 218, 438, 237, 457, 44, 19, 274]  # Nose landmarks

        self.MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39,
                                37]  # Mouth landmarks




        self.camera = camera
        self.face_box_location = None

        self.detect_face_thread = threading.Thread(target=self.detect_face, daemon=True)
        self.face_mesh_thread = threading.Thread(target=self.findMeshInFace, daemon=True)

    def detect_face(self):
        """Continuously detects face in the camera feed."""
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

    def findMeshInFace(self):
        # Initialize a dictionary to store the landmarks for facial features
        landmarks = {}

        # Convert the input image to RGB as Mediapipe expects RGB images
        img = self.camera.feed
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to find face landmarks using the FaceMesh model
        results = self.faceMesh.process(img_rgb)
        # Check if any faces were detected
        if results.multi_face_landmarks:
            # Iterate over detected faces (here, max_num_faces = 1, so usually one face)
            for faceLms in results.multi_face_landmarks:
                # Initialize lists in the landmarks dictionary to store each facial feature's coordinates
                landmarks["left_eye_landmarks"] = []
                landmarks["right_eye_landmarks"] = []
                landmarks["left_iris_landmarks"] = []
                landmarks["right_iris_landmarks"] = []
                landmarks["nose_landmarks"] = []
                landmarks["mouth_landmarks"] = []
                landmarks["all_landmarks"] = []  # Store all face landmarks for complete face mesh

                # Loop through all face landmarks
                for i, lm in enumerate(faceLms.landmark):
                    h, w, ic = img.shape  # Get image height, width, and channel count
                    x, y = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel values

                    # Store the coordinates of all landmarks
                    landmarks["all_landmarks"].append((x, y))

                    # Store specific feature landmarks based on the predefined indices
                    if i in self.LEFT_EYE_LANDMARKS:
                        landmarks["left_eye_landmarks"].append((x, y))  # Left eye
                    if i in self.RIGHT_EYE_LANDMARKS:
                        landmarks["right_eye_landmarks"].append((x, y))  # Right eye
                    if i in self.LEFT_IRIS_LANDMARKS:
                        landmarks["left_iris_landmarks"].append((x, y))  # Left iris
                    if i in self.RIGHT_IRIS_LANDMARKS:
                        landmarks["right_iris_landmarks"].append((x, y))  # Right iris
                    if i in self.NOSE_LANDMARKS:
                        landmarks["nose_landmarks"].append((x, y))  # Nose
                    if i in self.MOUTH_LANDMARKS:
                        landmarks["mouth_landmarks"].append((x, y))  # Mouth

        # Return the processed image and the dictionary of feature landmarks
        self.camera.face_landmarks = landmarks


    def stop(self):
        if self.detect_face_thread.is_alive():
            self.detect_face_thread.join()

        print("Detector successfully closed.")
