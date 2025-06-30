import pygame
import threading
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gaze_tracker import GazeTracker
from consts import *


class Calibration:
    def __init__(self, gaze_tracker: GazeTracker):
        self.gaze_tracker = gaze_tracker
        self.iris_data_flag = False

        self.background_color = black

        self.calibration_gui_thread = threading.Thread(target=self.start_calibration)
        self.exit_event = threading.Event()

    def interpolate(self, start, end, step, total_steps):
        return start + (end - start) * (step / total_steps)

    def draw_crosshair(self, surface, x, y, size=7, color=black):
        pygame.draw.line(surface, color, (x - size, y), (x + size, y), 5)
        pygame.draw.line(surface, color, (x, y - size), (x, y + size), 5)

    def shrink_circle_at(self, screen, x, y):
        self.iris_data_flag = True
        for step in range(collapse_steps + 1):
            shrinking_radius = int(self.interpolate(radius, 0, step, collapse_steps))
            screen.fill(white)

            # Permanent outer black border
            pygame.draw.circle(screen, black, (x, y), radius, 3)

            pygame.draw.circle(screen, black, (x, y), shrinking_radius + 2)  # Shrinking outer border
            pygame.draw.circle(screen, red, (x, y), shrinking_radius)

            self.draw_crosshair(screen, x, y)
            pygame.display.flip()
            time.sleep(collapse_time)
        self.iris_data_flag = False

    def start_calibration(self):
        # Calibration GUI
        pygame.init()
        self.gaze_tracker.screen_width = screen_width
        self.gaze_tracker.screen_height = screen_height
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Calibration Display")

        positions = self.gaze_tracker.screen_positions

        current_x, current_y = positions[0]

        # Display calibration button
        font = pygame.font.Font(None, 100)
        button_text = font.render("Calibration", True, white)
        button_rect = button_text.get_rect(center=(screen_width // 2, screen_height // 2))
        screen.fill(self.background_color)
        screen.blit(button_text, button_rect)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
                elif event.type == pygame.QUIT:
                    self.stop_calibration()
                    return

        # Background transition from black to white
        for step in range(transition_steps + 1):
            bg_color = (
                int(self.interpolate(black[0], white[0], step, transition_steps)),
                int(self.interpolate(black[1], white[1], step, transition_steps)),
                int(self.interpolate(black[2], white[2], step, transition_steps))
            )
            screen.fill(bg_color)
            pygame.display.flip()
            time.sleep(transition_time)

        for idx, (x, y) in enumerate(positions):
            for step in range(transition_steps + 1):
                intermediate_x = int(self.interpolate(current_x, x, step, transition_steps))
                intermediate_y = int(self.interpolate(current_y, y, step, transition_steps))

                screen.fill(white)
                pygame.draw.circle(screen, black, (intermediate_x, intermediate_y), radius + 3)
                pygame.draw.circle(screen, red, (intermediate_x, intermediate_y), radius)
                self.draw_crosshair(screen, intermediate_x, intermediate_y)

                pygame.display.flip()
                time.sleep(transition_time)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        self.stop_calibration()
                        return

            current_x, current_y = x, y

            # Use the shrink_circle_at method here
            if idx != 0:
                self.shrink_circle_at(screen, x, y)

            time.sleep(0.1)

        self.stop_calibration()

    def stop_calibration(self):
        self.exit_event.set()
        pygame.quit()
        print('Exiting Calibration')

# Debugging purposes
# if __name__ == "__main__":
#     calibration = Calibration()
#     calibration.start_calibration()
