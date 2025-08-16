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
                # self.image_preprocessing()            # <---------------- Ovde se vrsi obrada frejm po frejm
                self.show_in_window(win_name='Camera Feed', img=self.feed)

    def image_preprocessing(self):
        pass


    def get_feed(self):
        while self.running:
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


