import cv2
import cv2 as cv
import threading
import numpy as np


def show_in_window(winname, img, x = 0, y = 0):
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, x, y)   # Move it to (x,y)
    cv.imshow(winname,img)
    cv.waitKey(1)


class Camera:

    def __init__(self):

        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Cannot open camera. Make sure camera is connected properly.")
            exit()
        else:
            print("Camera setup successful.")

        success, self.feed = self.cap.read()

    def display_feed(self):
        while True:
            show_in_window('Camera Feed', self.feed)

    def edit_frame(self, frame):
        #frame = cv.resize(frame, (600, 600), interpolation=cv.INTER_NEAREST)
        frame = cv.flip(frame, 1)
        return frame

    def get_feed(self):

        while True:
            # Capture frame-by-frame
            success, self.feed = self.cap.read()
            self.feed = self.edit_frame(self.feed)
            if not success:
                continue

    def close_cam(self):
        self.cam.release()
        cv.destroyAllWindows()
        print("Camera successfully closed.")


# if __name__ == '__main__':
#     camera = Camera()
#
#     t1 = threading.Thread(target=camera.get_feed, args=())
#     t1.setDaemon(True)
#     t1.start()
#
#     t2 = threading.Thread(target=camera.display_feed(), args=())
#     t2.setDaemon(True)
#     t2.start()

