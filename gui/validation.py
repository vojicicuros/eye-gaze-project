import pygame
import threading
import time
from src.gaze_tracker import GazeTracker
from consts import *


class Validation:
    def __init__(self, gaze_tracker: GazeTracker):

        self.screen = None
        self.background_color = black

        self.gaze_tracker = gaze_tracker
        self.iris_data_flag = False
        self.positions = gaze_tracker.screen_positions

        self.validation_gui_thread = threading.Thread(target=self.start_validation)
        self.draw_gaze_gui_thread = threading.Thread(target=self.draw_gaze)

        self.exit_event = threading.Event()

    def interpolate(self, start, end, step, total_steps):
        return start + (end - start) * (step / total_steps)

    def draw_crosshair(self, surface, x, y, size=7, color=white):
        pygame.draw.line(surface, color, (x - size, y), (x + size, y), 5)
        pygame.draw.line(surface, color, (x, y - size), (x, y + size), 5)

    def shrink_circle_at(self, screen, x, y):
        self.iris_data_flag = True
        for step in range(collapse_steps + 1):
            shrinking_radius = int(self.interpolate(radius, 0, step, collapse_steps))
            self.screen.fill(black)

            # Permanent outer black border
            pygame.draw.circle(screen, white, (x, y), radius, 3)
            pygame.draw.circle(screen, white, (x, y), shrinking_radius + 2)  # Shrinking outer border
            pygame.draw.circle(screen, relaxing_green, (x, y), shrinking_radius)

            self.draw_crosshair(screen, x, y)
            pygame.display.flip()
            time.sleep(collapse_time)

        self.iris_data_flag = False

    def stop_validation(self):
        self.exit_event.set()
        pygame.quit()
        print('Exiting Validation')

    def start_validation(self):

        # Validation GUI
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Validation Display")

        current_x, current_y = self.positions[0]

        # Display validation button
        font = pygame.font.Font(None, 100)
        button_text = font.render("Validation", True, black)
        button_rect = button_text.get_rect(center=(screen_width // 2, screen_height // 2))
        self.screen.fill(white)
        self.screen.blit(button_text, button_rect)
        pygame.display.flip()
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    return

        # Background transition from white to black
        for step in range(transition_steps + 1):
            bg_color = (int(self.interpolate(white[0], black[0], step, transition_steps)),
                        int(self.interpolate(white[1], black[1], step, transition_steps)),
                        int(self.interpolate(white[2], black[2], step, transition_steps)))
            self.screen.fill(bg_color)
            pygame.display.flip()
            time.sleep(transition_time)

        for idx, (x, y) in enumerate(self.positions):
            for step in range(transition_steps + 1):
                intermediate_x = int(self.interpolate(current_x, x, step, transition_steps))
                intermediate_y = int(self.interpolate(current_y, y, step, transition_steps))

                self.screen.fill(black)
                pygame.draw.circle(self.screen, white, (intermediate_x, intermediate_y), radius + 3)
                pygame.draw.circle(self.screen, relaxing_green, (intermediate_x, intermediate_y), radius)
                self.draw_crosshair(self.screen, intermediate_x, intermediate_y)

                pygame.display.flip()
                time.sleep(transition_time)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                        self.stop_validation()
                        return

            current_x, current_y = x, y

            if idx != 0:
                self.shrink_circle_at(self.screen, x, y)

            time.sleep(0.1)
        self.stop_validation()

    def draw_gaze(self, alpha=0.2):
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

                # Draw main gaze dot (fully opaque red)
                dot_color = (*teal, 255)
                pygame.draw.circle(trail_surface, dot_color, (int(smoothed_pos[0]), int(smoothed_pos[1])), 10)

                # Blit to main screen
                # self.screen.fill(self.background_color)
                self.screen.blit(trail_surface, (0, 0))
                pygame.display.update()

            clock.tick(30)  # 30 FPS

