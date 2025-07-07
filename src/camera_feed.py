import cv2
import threading
from cam_config import *


class Camera:

    def __init__(self):
        self.landmarks_lock = threading.Lock()  # Create a lock

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_RES_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_RES_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for smoother output
        print(self.cap.get(cv2.CAP_PROP_FOURCC))
        if not self.cap.isOpened():
            print("Cannot open camera. Make sure camera is connected properly.")
            exit()
        else:
            print("Camera setup successful.")

        success, self.feed = self.cap.read()
        self.raw_feed = None
        self.running = True
        self.eye_box = None
        self.eyes_landmarks = None

        self.eye_box_initialized = False
        self.eye_crop_width = 100
        self.eye_crop_height = 50

        self.get_feed_thread = threading.Thread(target=self.get_feed, daemon=True)
        self.display_feed_thread = threading.Thread(target=self.display_feed, daemon=True)

    def draw_eyes_landmarks(self, img):

        # Mesh related
        keys_eyes = ["left_eye"]
        keys_iris = ["left_iris"]
        keys_iris_center = ["l_iris_center"]
        with self.landmarks_lock:
            if self.eyes_landmarks:
                for key in keys_eyes:
                    for landmark in self.eyes_landmarks[key]:
                        cv2.circle(img, (landmark[0], landmark[1]), 1, (0, 0, 255), 1)

                for key in keys_iris:
                    for landmark in self.eyes_landmarks[key]:
                        cv2.circle(img, (landmark[0], landmark[1]), 1, (255, 0, 0), 1)

                for key in keys_iris_center:
                    cv2.circle(img, (self.eyes_landmarks[key][0], self.eyes_landmarks[key][1]), 1, (255, 255, 255), 1)

    def show_in_window(self, win_name, img):
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, x=0, y=0)

        self.draw_eyes_landmarks(img)

        # Resize image for display only
        display_img = cv2.resize(img, (640, 360))  # Adjust size as needed

        cv2.imshow(win_name, display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def display_feed(self):
        while self.running:
            if self.feed is not None:
                # self.image_preprocessing()
                self.show_in_window(win_name='Camera Feed', img=self.feed)

    def image_preprocessing(self):

        with self.landmarks_lock:
            iris_center = self.eyes_landmarks.get("l_iris_center") if self.eyes_landmarks else None
            eye_landmarks = self.eyes_landmarks.get("left_eye") if self.eyes_landmarks else None

        # Calculate crop box size only once, from eye landmarks
        if not self.eye_box_initialized and eye_landmarks is not None:
            xs = [int(p[0]) for p in eye_landmarks]
            ys = [int(p[1]) for p in eye_landmarks]
            self.eye_crop_width = max(xs) - min(xs) + 10
            self.eye_crop_height = max(ys) - min(ys) + 10
            self.eye_box_initialized = True

        # Crop around iris center with fixed size
        if iris_center and self.eye_box_initialized:
            x_center, y_center = iris_center

            x1 = max(0, x_center - self.eye_crop_width // 2)
            y1 = max(0, y_center - self.eye_crop_height // 2)
            x2 = min(self.feed.shape[1], x_center + self.eye_crop_width // 2)
            y2 = min(self.feed.shape[0], y_center + self.eye_crop_height // 2)

            cv2.rectangle(self.feed, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # if y2 > y1 and x2 > x1:
            #     self.feed = self.feed[y1:y2, x1:x2]  # Crop feed
            # else:
            #     print(f"Invalid crop area: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    def get_feed(self):
        while self.running:
            # Capture frame-by-frame
            success, self.raw_feed = self.cap.read()
            if not success:
                continue

            self.feed = self.raw_feed.copy()

    def stop(self):
        self.running = False
        self.cap.release()
        print("Camera successfully closed.")

        # Ending all threads
        if self.get_feed_thread.is_alive():
            self.get_feed_thread.join()
        if self.display_feed_thread.is_alive():
            self.display_feed_thread.join()


