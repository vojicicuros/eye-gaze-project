import cv2
import threading


class Camera:

    def __init__(self):

        self.landmarks_lock = threading.Lock()  # Create a lock

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Cannot open camera. Make sure camera is connected properly.")
            exit()
        else:
            print("Camera setup successful.")

        success, self.feed = self.cap.read()
        self.running = True
        self.face_box = None
        self.face_landmarks = None

        self.get_feed_thread = threading.Thread(target=self.get_feed, daemon=True)
        self.display_feed_thread = threading.Thread(target=self.display_feed, daemon=True)  # Start thread within class

    def show_in_window(self, win_name, img):
        if self.face_box:
            h, w, _ = img.shape  # Get frame dimensions
            # Convert relative bbox coordinates to pixel values
            x, y, w_box, h_box = (
                int(self.face_box.xmin * w),
                int(self.face_box.ymin * h),
                int(self.face_box.width * w),
                int(self.face_box.height * h)
            )
            # Draw a blue rectangle around the face
            cv2.rectangle(img, (x, y), (x + w_box, y + h_box), (0, 255, 0), 1)
        #Mesh related
        keys_eyes = ["left_eye", "right_eye"]
        keys_iris = ["left_iris", "right_iris"]
        with self.landmarks_lock:
            if self.face_landmarks:
                for key in keys_eyes:
                    for landmark in self.face_landmarks[key]:
                        cv2.circle(img, (landmark[0], landmark[1]), 1, (255, 0, 255), 1)
                for key in keys_iris:
                    for landmark in self.face_landmarks[key]:
                        cv2.circle(img, (landmark[0], landmark[1]), 1, (0, 255, 0), 1)

        cv2.namedWindow(win_name)  # Create a named window
        cv2.moveWindow(win_name, x=0, y=0)  # Move it to (x,y)
        cv2.imshow(win_name, img)
        # Resetting face_box and face_landmarks
        self.face_box = None

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop()

    def display_feed(self, face_box=None):
        while self.running:
            if self.feed is not None:
                self.show_in_window(win_name='Camera Feed', img=self.feed)


    def edit_frame(self, frame):
        #frame = cv2.flip(frame, 1)
        return frame

    def get_feed(self):

        while self.running:
            # Capture frame-by-frame
            success, self.feed = self.cap.read()
            self.feed = self.edit_frame(self.feed)
            if not success:
                continue

    def stop(self):
        self.running = False
        self.cap.release()
        print("Camera successfully closed.")

        # Ending all threads
        if self.get_feed_thread.is_alive():
            self.get_feed_thread.join()
        if self.display_feed_thread.is_alive():
            self.display_feed_thread.join()


