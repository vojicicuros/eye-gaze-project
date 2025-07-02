import pygame
import threading
import time
from src.gaze_tracker import GazeTracker
from consts import *


class Gazing:
    def __init__(self, gaze_tracker: GazeTracker):

        self.gaze_tracker = gaze_tracker

        self.screen = None
        self.background_color = black

        self.gazing_gui_thread = threading.Thread(target=self.start_gaze_part)
        self.gazing_data_thread = threading.Thread(target=self.get_and_estimate_data)
        self.draw_gaze_thread = threading.Thread(target=self.draw_gaze)

        self.exit_event = threading.Event()

    def start_gaze_part(self):

        # Gazing GUI
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Gazing Display")
        self.screen.fill(self.background_color)
        pygame.display.flip()

        waiting = True
        while waiting:

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
                elif event.type == pygame.QUIT:
                    self.stop_gazing()
                    return

        self.stop_gazing()

    def get_and_estimate_data(self):
        while not self.exit_event.is_set():
            l_iris_cent = self.gaze_tracker.detector.camera.eyes_landmarks.get("l_iris_center")
            self.gaze_tracker.gaze = self.gaze_tracker.linear_estimation(l_iris_cent)

    def draw_gaze(self):
        clock = pygame.time.Clock()
        while not self.exit_event.is_set():
            if self.screen and self.gaze_tracker.gaze is not None:
                # Clear screen before drawing new circle
                self.screen.fill(self.background_color)

                x, y = map(int, self.gaze_tracker.gaze)
                pygame.draw.circle(self.screen, red, (x, y), 10)

                pygame.display.update()

            clock.tick(30)  # 30 FPS

    def stop_gazing(self):
        self.exit_event.set()
        pygame.quit()
        print('Exiting Gazing')
