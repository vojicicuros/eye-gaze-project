import pygame
import threading
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

        #self.stop_gazing()

    def get_and_estimate_data(self):
        while not self.exit_event.is_set():
            eye_input = self.gaze_tracker.detector.camera.eyes_landmarks.get("l_iris_center")

            if method_num == 0:
                self.gaze_tracker.gaze = self.gaze_tracker.linear_mapping(eye_input)
            elif method_num == 1:
                self.gaze_tracker.gaze = self.gaze_tracker.polynomial_mapping(eye_input)
            elif method_num == 2:
                self.gaze_tracker.gaze = self.gaze_tracker.svr_mapping(eye_input)

    def draw_gaze(self, alpha=0.5):
        # Smoothing factor (lower = smoother)

        clock = pygame.time.Clock()
        smoothed_pos = None
        trail_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

        while not self.exit_event.is_set():
            if self.screen and self.gaze_tracker.gaze is not None:
                # Smooth the gaze position
                x, y = self.gaze_tracker.gaze
                if smoothed_pos is None:
                    smoothed_pos = [x, y]
                else:
                    smoothed_pos[0] = alpha * x + (1 - alpha) * smoothed_pos[0]
                    smoothed_pos[1] = alpha * y + (1 - alpha) * smoothed_pos[1]

                # Fade previous trails
                trail_surface.fill((0, 0, 0, 35), special_flags=pygame.BLEND_RGBA_SUB)

                # # Draw halo (larger, transparent)
                # halo_color = (*red, 60)  # RGBA: semi-transparent red
                # pygame.draw.circle(trail_surface, halo_color, (int(smoothed_pos[0]), int(smoothed_pos[1])), 25)

                # Draw main gaze dot (fully opaque red)
                dot_color = (*red, 255)
                pygame.draw.circle(trail_surface, dot_color, (int(smoothed_pos[0]), int(smoothed_pos[1])), 10)

                # Blit to main screen
                self.screen.fill(self.background_color)
                self.screen.blit(trail_surface, (0, 0))
                pygame.display.update()

            clock.tick(30)  # 30 FPS

    def stop_gazing(self):
        self.exit_event.set()
        pygame.quit()
        print('Exiting Gazing')
