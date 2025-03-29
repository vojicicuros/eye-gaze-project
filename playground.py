import mediapipe as mp
import cv2


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

    def findMeshInFace(self, img):
        landmarks = {"left_eye": [], "right_eye": [], "left_iris": [], "right_iris": []}
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
        for key in landmarks_eyes:
            for landmark in landmarks[key]:
                cv2.circle(image, (landmark[0], landmark[1]), 1, (255, 0, 255), 1)
        for key in landmarks_iris:
            for landmark in landmarks[key]:
                cv2.circle(image, (landmark[0], landmark[1]), 1, (0, 255, 0), 1)

        cv2.namedWindow("Live feed")  # Create a named window
        cv2.moveWindow("Live feed", x=0, y=0)  # Move it to (x,y)
        cv2.imshow("Live feed",image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
