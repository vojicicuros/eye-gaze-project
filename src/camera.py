import cv2 as cv
import numpy as np
import threading


def show_in_window(winname, img, x = 0, y = 0):
    cv.namedWindow(winname)        # Create a named window
    cv.moveWindow(winname, x, y)   # Move it to (x,y)
    cv.imshow(winname,img)


class Camera:

    def __init__(self):

        self.cam = self.cam_setup()

        start_recording = threading.Thread(target=self.get_feed, args=())
        start_recording.start()

        self.feed = None

    def get_feed(self):
        while True:
            # Capture frame-by-frame
            success, self.feed = self.cam.read()
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            if cv.waitKey(1) == ord('q'):
                self.close_cam()
                break

    def display_feed(self):
        while True:
            frame = cv.flip(self.feed, 1)
            show_in_window('Camera Feed', frame)
            if cv.waitKey(1) == ord('q'):
                self.close_cam()
                break

    def cam_setup(self):

        cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        else:
            print("Camera setup successful.")
        return cap

    def close_cam(self):
        self.cam.release()
        cv.destroyAllWindows()
        print("Camera successfully closed.")



if __name__ == '__main__':
    camera = Camera()