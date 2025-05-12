import mediapipe as mp
import cv2

"""
This file is used for adjusting eye landmark values inside of FaceMeshDetector class
Testing purposes only.
"""

class FaceMeshDetector:

    def __init__(self):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                                 min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                   380, 381, 382, 362]
        self.RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                    144, 163, 7]
        self.LEFT_IRIS_LANDMARKS = [474, 475, 477, 476]
        self.RIGHT_IRIS_LANDMARKS = [469, 470, 471, 472]

    def iris_center(self, iris_landmarks):
        x1, y1 = iris_landmarks[0]
        x2, y2 = iris_landmarks[1]
        x3, y3 = iris_landmarks[2]
        x4, y4 = iris_landmarks[3]

        # Koeficijenti pravaca
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

    def findMeshInFace(self, img):
        landmarks = {"left_eye": [], "right_eye": [], "left_iris": [], "right_iris": [],
                     "l_iris_center": [], "r_iris_center": []}
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                for i, lm in enumerate(faceLms.landmark):
                    h, w, _ = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)

                    if i in self.LEFT_EYE_LANDMARKS:
                        landmarks["left_eye"].append((x, y))
                    if i in self.RIGHT_EYE_LANDMARKS:
                        landmarks["right_eye"].append((x, y))
                    if i in self.LEFT_IRIS_LANDMARKS:
                        landmarks["left_iris"].append((x, y))
                    if i in self.RIGHT_IRIS_LANDMARKS:
                        landmarks["right_iris"].append((x, y))

        # with self.camera.landmarks_lock:
        landmarks["l_iris_center"] = self.iris_center(landmarks["left_iris"])
        landmarks["r_iris_center"] = self.iris_center(landmarks["right_iris"])

        return img, landmarks


if __name__ == '__main__':
    detector = FaceMeshDetector()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


    while True:
        success, image = cap.read()
        if not success:
            break

        # height, width = image.shape[:2]
        # image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        image, landmarks = detector.findMeshInFace(image)
        landmarks_eyes = ["left_eye", "right_eye"]
        landmarks_iris = ["left_iris", "right_iris"]
        landmarks_iris_center = ["l_iris_center", "r_iris_center"]

        for key in landmarks_eyes:
            for landmark in landmarks[key]:
                cv2.circle(image, (landmark[0], landmark[1]), 1, (255, 0, 0), 1)
        for key in landmarks_iris:
            for landmark in landmarks[key]:
                cv2.circle(image, (landmark[0], landmark[1]), 1, (0, 255, 0), 1)
        for key in landmarks_iris_center:
            cv2.circle(image, (landmarks[key][0], landmarks[key][1]), 1, (0, 0, 255), 1)

        cv2.namedWindow("Live feed")  # Create a named window
        cv2.moveWindow("Live feed", x=0, y=0)  # Move it to (x,y)
        cv2.imshow("Live feed",image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
