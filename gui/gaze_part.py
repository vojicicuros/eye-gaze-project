import time
import pygame
import threading
from src.gaze_tracker import GazeTracker
from consts import *

class Gazing:
    def __init__(self, gaze_tracker: GazeTracker):
        self.gaze_tracker = gaze_tracker

        self.screen = None
        self.background_color = black

        # GUI loop (pygame)
        self.gazing_gui_thread = threading.Thread(target=self.start_gaze_part, daemon=True)
        # Procena pogleda (linear/poly/SVM) u odvojenoj niti
        self.gazing_data_thread = threading.Thread(target=self.get_and_estimate_data, daemon=True)

        self.draw_gaze_thread = threading.Thread(target=self.draw_gaze, daemon=True)

        self.exit_event = threading.Event()

    # Glavni GUI
    def start_gaze_part(self, alpha: float = 0.25, dot_radius: int = 10):
        """
        Jedan GUI/event loop sa trail-om i smoothingom.
        - alpha: 0..1  (manje = jače zaglađivanje)
        - dot_radius: poluprečnik glavne tačke
        Prečice:
          [  smanji alpha (više zaglađivanja)
          ]  poveća alpha (manje zaglađivanja)
          -  smanji fade (duži trag)
          =  poveća fade (kraći trag)
        """
        import pygame
        pygame.init()
        pygame.display.init()

        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Gazing Display")

        clock = pygame.time.Clock()

        # Trail površina (transparentna)
        trail_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

        # Kontrole traga
        trail_fade = 35  # veća vrednost = brže bledi trag (0..255)
        trail_dot_alpha = 140  # alfa tačke za trag (manje od glavne tačke)
        main_dot_alpha = 255  # glavna tačka potpuno vidljiva

        # Teal za glavnu (kao u validaciji), trail malo blaži
        trail_color = (*teal, trail_dot_alpha)
        main_color = (*teal, main_dot_alpha)

        smoothed_pos = None

        if not self.gazing_data_thread.is_alive():
            try:
                self.gazing_data_thread.start()
            except RuntimeError:
                pass

        running = True
        while running and not self.exit_event.is_set():
            # 1) Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_gazing()
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.stop_gazing()
                        running = False
                    elif event.key == pygame.K_LEFTBRACKET:  # [
                        alpha = max(0.01, alpha - 0.05)
                        print(f"smoothing alpha -> {alpha:.2f} (manje = više zaglađeno)")
                    elif event.key == pygame.K_RIGHTBRACKET:  # ]
                        alpha = min(0.99, alpha + 0.05)
                        print(f"smoothing alpha -> {alpha:.2f} (veće = manje zaglađeno)")
                    elif event.key == pygame.K_MINUS:  # -
                        trail_fade = max(0, trail_fade - 5)
                        print(f"trail fade -> {trail_fade} (manje = duži trag)")
                    elif event.key == pygame.K_EQUALS:  # =
                        trail_fade = min(255, trail_fade + 5)
                        print(f"trail fade -> {trail_fade} (veće = kraći trag)")

            pygame.event.pump()

            # Pozadina (po želji jednobojna ili ostavi prethodni frame)
            self.screen.fill(self.background_color)

            # Trail bledi vremenom
            # Subtrakcija male vrednosti iz alfa kanala pravi “vapor trail” efekat
            trail_surface.fill((0, 0, 0, trail_fade), special_flags=pygame.BLEND_RGBA_SUB)

            # Ako imamo novu procenu pogleda – prikaži
            if self.gaze_tracker.gaze is not None:
                x, y = self.gaze_tracker.gaze

                # Exponential smoothing
                if smoothed_pos is None:
                    smoothed_pos = [x, y]
                else:
                    smoothed_pos[0] = alpha * x + (1 - alpha) * smoothed_pos[0]
                    smoothed_pos[1] = alpha * y + (1 - alpha) * smoothed_pos[1]

                ix, iy = int(smoothed_pos[0]), int(smoothed_pos[1])

                # Nacrtaj TRAIL tačku (blaža, poluprozirna) na trail_surface
                pygame.draw.circle(trail_surface, trail_color, (ix, iy), dot_radius)

                # Zalepi trail na ekran
                self.screen.blit(trail_surface, (0, 0))

                # Nacrtaj GLAVNU tačku (izraženija) direktno na screen
                pygame.draw.circle(self.screen, main_color, (ix, iy), dot_radius)

            # Swap
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        print('Exiting Gazing (GUI)')

    # DATA THREAD: čita landmarke i radi PREDICT
    def get_and_estimate_data(self):
        """
        Brza, robusna petlja:
        - uzima snapshot landmarka,
        - radi SAMO predict (nema treniranja niti disk I/O),
        - ima guard-e na None/prazne ulaze,
        - štiti se od tihih izuzetaka i ne guši CPU.
        """
        from queue import Empty

        # Ako imaš internu Queue(maxsize=1) između face_mesh_thread i ovoga, koristi je.
        # Ako nemaš, uzmi snapshot iz shared strukture sa guardovima:
        while not self.exit_event.is_set():
            t0 = time.perf_counter()
            try:
                # --- SNAPSHOT LANDMARKA (bez blokiranja glavnog GUI threada) ---
                # Ako imaš lock u camera feed-u (npr. self.gaze_tracker.cam.landmarks_lock),
                # ovde ga iskoristi sa timeout-om; u suprotnom, čitaj direktno uz guard.
                lm = None
                try:
                    lm = self.gaze_tracker.cam.eyes_landmarks  # dict ili None
                except Exception:
                    lm = None

                eye_center_input = None
                eye_outer_input = None
                if lm:
                    eye_center_input = lm.get("l_iris_center")
                    eye_outer_input = lm.get("left_eye")

                # Izračunaj uglove oka, ali robustno na prazne liste
                l_corner_input = r_corner_input = None
                try:
                    l_corner_input, r_corner_input = self.gaze_tracker.detector.get_eye_corners(eye_outer_input)
                except Exception:
                    l_corner_input, r_corner_input = None, None

                # --- PREDICT (bez treniranja i I/O) ---
                method = method_num  # iz consts; 0=linear,1=poly,2=svr

                if method == 0:
                    if eye_center_input is None:
                        time.sleep(0.001); continue
                    gaze = self.gaze_tracker.linear_mapping(eye_center_input)
                elif method == 1:
                    if eye_center_input is None or l_corner_input is None or r_corner_input is None:
                        time.sleep(0.001); continue
                    # OVO MORA BITI "predict" grana (bez recompute/training/load)
                    gaze = self.gaze_tracker.polynomial_mapping(eye_center_input,
                                                                l_corner_input,
                                                                r_corner_input)
                else:
                    if eye_center_input is None or l_corner_input is None or r_corner_input is None:
                        time.sleep(0.001); continue
                    # OVO MORA BITI "predict" grana (bez training/load po frejmu)
                    gaze = self.gaze_tracker.svr_mapping(eye_center_input,
                                                         l_corner_input,
                                                         r_corner_input)

                if gaze is not None:
                    # osiguraj se da su floatovi
                    self.gaze_tracker.gaze = (float(gaze[0]), float(gaze[1]))

            except Exception as e:
                # Loguj, ali ne ruši nit
                import traceback
                print("[Gazing.get_and_estimate_data] exception:\n", traceback.format_exc())
                time.sleep(0.01)

            # Mikro “disanje” da ne guši CPU ako je sve već urađeno
            dt = time.perf_counter() - t0
            if dt < 0.001:
                time.sleep(0.001)

    def draw_gaze(self):
        """
        Ova nit je sada namerno prazna (NO-OP), jer se sav pygame render i event loop
        odvija u start_gaze_part(). Ostavili smo je da main.py ne puca ako je pokrene.
        """
        while not self.exit_event.is_set():
            time.sleep(0.05)

    def stop_gazing(self):
        self.exit_event.set()

        # Join niti, ali sa timeout-om da se ne zaglaviš
        try:
            if self.gazing_data_thread.is_alive():
                self.gazing_data_thread.join(timeout=1.0)
        except Exception:
            pass

        try:
            if self.draw_gaze_thread.is_alive():
                self.draw_gaze_thread.join(timeout=1.0)
        except Exception:
            pass

        # GUI thread će izaći preko event loop-a i pygame.quit() u start_gaze_part()
        print('Exiting Gazing')
