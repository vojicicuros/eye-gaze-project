import pygame
import threading
import time

def interpolate(start, end, step, total_steps):
    return start + (end - start) * (step / total_steps)

def draw_crosshair(surface, x, y, size=7, color=(0, 0, 0)):
    pygame.draw.line(surface, color, (x - size, y), (x + size, y), 5)
    pygame.draw.line(surface, color, (x, y - size), (x, y + size), 5)

def shrink_circle_at(screen, x, y, radius, collapse_steps, collapse_time, white, red, black):
    for step in range(collapse_steps + 1):
        shrinking_radius = int(interpolate(radius, 0, step, collapse_steps))
        screen.fill(white)

        # Permanent outer black border
        pygame.draw.circle(screen, black, (x, y), radius, 3)

        pygame.draw.circle(screen, black, (x, y), shrinking_radius + 2)  # Shrinking outer border
        pygame.draw.circle(screen, red, (x, y), shrinking_radius)

        draw_crosshair(screen, x, y)
        pygame.display.flip()
        time.sleep(collapse_time)

def graphics():
    pygame.init()
    info = pygame.display.Info()
    screen_width, screen_height = info.current_w, info.current_h
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)
    pygame.display.set_caption("Calibration Display")

    black = (0, 0, 0)
    red = (255, 0, 0)
    white = (255, 255, 255)

    radius = 20
    padding = 50
    transition_steps = 15
    transition_time = 0.02
    collapse_steps = 20
    collapse_time = 0.05

    # Define 16 circle positions (4 in each row)
    row_step = (screen_height - 2 * padding) // 3
    col_step = (screen_width - 2 * padding) // 3

    positions = [(screen_width // 2, screen_height // 2)] + [
        (padding + i * col_step, padding + j * row_step)
        for j in range(4) for i in range(4)
    ]

    print("Circle Positions:")
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
            int(interpolate(black[0], white[0], step, transition_steps)),
            int(interpolate(black[1], white[1], step, transition_steps)),
            int(interpolate(black[2], white[2], step, transition_steps))
        )
        screen.fill(bg_color)
        pygame.display.flip()
        time.sleep(transition_time)

    for idx, (x, y) in enumerate(positions):
        print(f"Circle {idx + 1}: ({x}, {y})")
        for step in range(transition_steps + 1):
            intermediate_x = int(interpolate(current_x, x, step, transition_steps))
            intermediate_y = int(interpolate(current_y, y, step, transition_steps))

            screen.fill(white)
            pygame.draw.circle(screen, black, (intermediate_x, intermediate_y), radius + 3)
            pygame.draw.circle(screen, red, (intermediate_x, intermediate_y), radius)
            draw_crosshair(screen, intermediate_x, intermediate_y)

            pygame.display.flip()
            time.sleep(transition_time)

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return

        current_x, current_y = x, y

        # Use the shrink_circle_at method here
        shrink_circle_at(screen, x, y, radius, collapse_steps, collapse_time, white, red, black)

        time.sleep(0.1)

    pygame.quit()


if __name__ == "__main__":
    graphics()
