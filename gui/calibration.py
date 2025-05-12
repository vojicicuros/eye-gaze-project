import json

import numpy as np
import pygame
import threading
import time
import sys
import os
from src.gaze_tracker import GazeTracker

# constants
black = (0, 0, 0)
red = (255, 0, 0)
white = (255, 255, 255)
radius = 20
padding = 50
transition_steps = 15
transition_time = 0.02
collapse_steps = 20
collapse_time = 0.05
num_of_dots = 3


class Calibration:
    def __init__(self, gaze_tracker: GazeTracker):
        self.gaze_tracker = gaze_tracker
        self.iris_data_flag = False
        self.screen_positions = None

        self.start_calibration_thread = threading.Thread(target=self.start_calibration)
        self.iris_data_thread = threading.Thread(target=self.collect_iris_data)
        self.iris_data_thread.daemon = True  # Daemon thread will exit when the main program exits
        self.exit_event = threading.Event()

    def collect_iris_data(self):
        iris_data_dict = {
            "l_iris_boundaries": [],
            "r_iris_boundaries": [],
            "l_iris_cent": [],
            "r_iris_cent": [],
            "screen_position": []
        }
        l_iris_data = []
        r_iris_data = []

        was_collecting = False  # Tracks the previous state of the flag

        while not self.exit_event.is_set():
            if self.iris_data_flag:
                was_collecting = True
                l_iris_bound = self.gaze_tracker.detector.camera.eyes_landmarks.get("left_iris")
                r_iris_bound = self.gaze_tracker.detector.camera.eyes_landmarks.get("right_iris")
                l_iris_cent = self.gaze_tracker.detector.camera.eyes_landmarks.get("l_iris_center")
                r_iris_cent = self.gaze_tracker.detector.camera.eyes_landmarks.get("r_iris_center")

                if l_iris_cent is not None:
                    l_iris_data.append(l_iris_cent)
                if r_iris_cent is not None:
                    r_iris_data.append(r_iris_cent)

            else:
                # Only append once when iris_data_flag switches from True to False
                if was_collecting:
                    if l_iris_data and r_iris_data:
                        iris_data_dict["l_iris_cent"].append(np.median(l_iris_data, axis=0).astype(int).tolist())
                        iris_data_dict["r_iris_cent"].append(np.median(r_iris_data, axis=0).astype(int).tolist())
                    l_iris_data = []
                    r_iris_data = []
                    was_collecting = False  # Reset flag to prevent repeated appending

            time.sleep(0.01)
        iris_data_dict["screen_position"] = self.screen_positions[1:]
        iris_data_list = []
        for l_iris, r_iris, screen_pos in zip(
                iris_data_dict["l_iris_cent"],
                iris_data_dict["r_iris_cent"],
                self.screen_positions[1:]
        ):
            iris_data_list.append({
                "l_iris_cent": l_iris,
                "r_iris_cent": r_iris,
                "screen_position": screen_pos
            })

        self.save_iris_data(data=iris_data_list)

    def save_iris_data(self, data):
        file_path = os.path.join("data", "iris_data.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []

        existing_data.extend(data)
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)
            print("Saved data into .json file.")

    def interpolate(self, start, end, step, total_steps):
        return start + (end - start) * (step / total_steps)

    def draw_crosshair(self, surface, x, y, size=7, color=(0, 0, 0)):
        pygame.draw.line(surface, color, (x - size, y), (x + size, y), 5)
        pygame.draw.line(surface, color, (x, y - size), (x, y + size), 5)

    def shrink_circle_at(self, screen, x, y, radius, collapse_steps, collapse_time, white, red, black):
        start_time = time.time()
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
        end_time = time.time()
        #print(f"Shrinking circle at ({x},{y}) - time: {end_time-start_time}")

    def calculate_positions(self, screen_height, screen_width, num_of_dots):
        n = num_of_dots

        row_step = (screen_height - 2 * padding) // (n-1)
        col_step = (screen_width - 2 * padding) // (n-1)

        positions = [(screen_width // 2, screen_height // 2)] + [
            (padding + i * col_step, padding + j * row_step)
            for j in range(n) for i in range(n)
        ]
        return positions

    def stop_calibration(self):
        self.exit_event.set()
        self.iris_data_thread.join()
        pygame.quit()
        print('Exiting Calibration')

    def start_calibration(self):
        # Calibration GUI + location logic
        pygame.init()
        info = pygame.display.Info()
        screen_width, screen_height = info.current_w, info.current_h
        screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Calibration Display")

        positions = self.calculate_positions(screen_height, screen_width, num_of_dots)
        self.screen_positions = positions.copy()

        print("Spot Positions:")
        current_x, current_y = positions[0]

        # Display calibration button
        font = pygame.font.Font(None, 100)
        button_text = font.render("Calibration", True, white)
        button_rect = button_text.get_rect(center=(screen_width // 2, screen_height // 2))
        screen.fill(black)
        screen.blit(button_text, button_rect)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
                elif event.type == pygame.QUIT:
                    pygame.quit()
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
                self.shrink_circle_at(screen, x, y, radius, collapse_steps, collapse_time, white, red, black)

            time.sleep(0.1)
        self.stop_calibration()


# Debugging purposes
# if __name__ == "__main__":
#     calibration = Calibration()
#     calibration.start_calibration()
