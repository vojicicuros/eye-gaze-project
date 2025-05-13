import pygame
import time


class GazeGUIBase:
    def __init__(self, gaze_tracker):
        self.gaze_tracker = gaze_tracker

    def interpolate(self, start, end, step, total_steps):
        return start + (end - start) * (step / total_steps)

    def draw_crosshair(self, surface, x, y, size=7, color=(0, 0, 0)):
        pygame.draw.line(surface, color, (x - size, y), (x + size, y), 5)
        pygame.draw.line(surface, color, (x, y - size), (x, y + size), 5)

    def shrink_circle_at(self, screen, x, y, radius, collapse_steps, collapse_time, white, red, black):
        for step in range(collapse_steps + 1):
            shrinking_radius = int(self.interpolate(radius, 0, step, collapse_steps))
            screen.fill(white)

            pygame.draw.circle(screen, black, (x, y), radius, 3)
            pygame.draw.circle(screen, black, (x, y), shrinking_radius + 2)
            pygame.draw.circle(screen, red, (x, y), shrinking_radius)
            self.draw_crosshair(screen, x, y)

            pygame.display.flip()
            time.sleep(collapse_time)
