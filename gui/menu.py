import pygame
import button

pygame.init()

#create game window
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")

#game variables
game_paused = False
menu_state = "main"

#define fonts
font = pygame.font.SysFont("arialblack", 40)

#define colours
TEXT_COL = (255, 255, 255)

#load button images
resume_img = pygame.image.load("images/button_resume.png").convert_alpha()
calibration_img = pygame.image.load("images/button_calibration.png").convert_alpha()
validation_img = pygame.image.load("images/button_validation.png").convert_alpha()
eye_gaze_img = pygame.image.load("images/button_eye_gaze.png").convert_alpha()
quit_img = pygame.image.load('images/button_quit.png').convert_alpha()
back_img = pygame.image.load('images/button_back.png').convert_alpha()

#create button instances
resume_button = button.Button(0, 0, resume_img, 1)
calibration_button = button.Button(304, 125, calibration_img, 1)
validation_button = button.Button(297, 250, validation_img, 1)
eye_gaze_button = button.Button(336, 375, eye_gaze_img, 1)
quit_button = button.Button(332, 450, quit_img, 1)

def draw_text(text, font, text_col, x, y):
  img = font.render(text, True, text_col)
  screen.blit(img, (x, y))

if __name__ == "__main__":
    #game loop
    run = True
    while run:

      screen.fill((52, 78, 91))

      #check if game is paused
      if game_paused == True:
        #check menu state
        if menu_state == "main":
          #draw pause screen buttons
          if resume_button.draw(screen):
            game_paused = False
          if calibration_button.draw(screen):
            game_paused = False
          if validation_button.draw(screen):
            game_paused = False
          if eye_gaze_button.draw(screen):
            game_paused = False
          if quit_button.draw(screen):
            run = False
      else:
        draw_text("Press SPACE to pause", font, TEXT_COL, 160, 250)

      #event handler
      for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
          if event.key == pygame.K_SPACE:
            game_paused = True
        if event.type == pygame.QUIT:
          run = False

      pygame.display.update()

    pygame.quit()