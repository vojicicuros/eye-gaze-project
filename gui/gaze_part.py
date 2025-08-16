import os, time, math, csv, threading
from collections import deque
from datetime import datetime

import pygame
from src.gaze_tracker import GazeTracker
from consts import *

try:
    # matplotlib i PIL samo za eksport (van real-time petlje)
    from PIL import Image
    import matplotlib.pyplot as plt
except Exception:
    # dozvoljavamo da se GUI pokrene i bez njih, ali eksport neće raditi
    Image = None
    plt = None


class Gazing:
    """
    Segment 'gaze_part':
    - renderuje pozadinski tekst (sliku) i tačku pogleda u realnom vremenu
    - beleži sve uzorke (t, x, y) u thread-safe bafer
    - na 'Q' pravi plot (pozadina + scatter) i CSV dump
    """
    def __init__(self, gaze_tracker: GazeTracker, bg_path: str = "text2.png", out_dir: str = "results"):
        self.gaze_tracker = gaze_tracker

        # --- GUI state ---
        self.screen = None
        self.bg_surface = None
        self.background_color = black
        self.bg_path = bg_path

        # --- Threads ---
        self.gazing_gui_thread = threading.Thread(target=self.start_gaze_part, daemon=True)
        self.gazing_data_thread = threading.Thread(target=self.get_and_estimate_data, daemon=True)
        self.draw_gaze_thread = threading.Thread(target=self.draw_gaze, daemon=True)

        self.exit_event = threading.Event()

        # --- Session buffers (thread-safe) ---
        self.buf_lock = threading.Lock()
        self.samples = deque(maxlen=200000)  # (t, x, y)
        self.session_start_ts = None
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        # render parametri
        self.alpha_default = 0.25
        self.dot_radius_default = 10

    # ---------- SESSION UTILS ----------

    def _session_reset(self):
        with self.buf_lock:
            self.samples.clear()
        self.session_start_ts = time.time()

    def _record_gaze_point(self, x: float, y: float):
        """Thread-safe upis jednog uzorka."""
        if self.session_start_ts is None:
            return
        ts = time.time() - self.session_start_ts
        with self.buf_lock:
            self.samples.append((ts, float(x), float(y)))

    def _compose_paths(self, method_label: str):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"reading_{method_label}_{stamp}"
        png_path = os.path.join(self.out_dir, base + ".png")
        csv_path = os.path.join(self.out_dir, base + ".csv")
        return png_path, csv_path

    def _export_csv(self, csv_path: str):
        with self.buf_lock:
            rows = list(self.samples)
        if not rows:
            print("[Gazing] Nema uzoraka za CSV.")
            return
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec", "x_px", "y_px"])
            w.writerows(rows)
        print(f"[Gazing] CSV sačuvan: {csv_path}")

    def _export_plot(self, png_path: str):
        """Crtaj pozadinu i scatter tačaka (bez blokiranja GUI niti)."""
        if Image is None or plt is None:
            print("[Gazing] matplotlib/Pillow nisu dostupni – preskačem plot.")
            return

        # 1) priprema podataka
        with self.buf_lock:
            rows = list(self.samples)
        if not rows:
            print("[Gazing] Nema uzoraka za plot.")
            return

        xs = [r[1] for r in rows]
        ys = [r[2] for r in rows]

        # 2) učitaj i poravnaj pozadinu
        bg_img = None
        try:
            bg_img = Image.open(self.bg_path).convert("RGBA").resize((screen_width, screen_height))
        except Exception as e:
            print(f"[Gazing] Ne mogu učitati pozadinu '{self.bg_path}': {e}")

        # 3) matplotlib figura u istim dimenzijama
        dpi = 100
        fig_w = screen_width / dpi
        fig_h = screen_height / dpi
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = plt.axes([0, 0, 1, 1])  # full-bleed
        ax.set_xlim([0, screen_width])
        ax.set_ylim([screen_height, 0])  # invert Y da se poklopi sa pygame koordinatama
        ax.axis("off")

        if bg_img is not None:
            ax.imshow(bg_img, extent=[0, screen_width, screen_height, 0])

        # 4) scatter (male, poluprozirne tačke)
        ax.scatter(xs, ys, s=8, alpha=0.5)

        fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print(f"[Gazing] Plot sačuvan: {png_path}")

    def _finalize_and_save(self, method_label: str):
        """Pozvati na izlazu: snimi CSV i PNG plot."""
        png_path, csv_path = self._compose_paths(method_label)
        self._export_csv(csv_path)
        self._export_plot(png_path)

    # ---------- RESOURCE LOADING ----------

    def _load_background_image(self, path: str = None):
        path = path or self.bg_path
        try:
            img = pygame.image.load(path)
            self.bg_surface = pygame.transform.smoothscale(img, (screen_width, screen_height)).convert()
            self.bg_path = path  # zapamti stvarno korišćenu pozadinu (za plot)
        except Exception as e:
            print(f"[Gazing] Nije moguće učitati '{path}': {e}. Koristim jednobojnu pozadinu.")
            self.bg_surface = pygame.Surface((screen_width, screen_height))
            self.bg_surface.fill((240, 240, 240))

    # ---------- GUI LOOP ----------

    def start_gaze_part(self, alpha: float = None, dot_radius: int = None):
        """Glavni GUI i prikupljanje tačaka. Na 'Q' eksportuje PNG + CSV."""
        alpha = self.alpha_default if alpha is None else alpha
        dot_radius = self.dot_radius_default if dot_radius is None else dot_radius

        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Gazing – Reading")

        # početni ekran
        self.screen.fill(white)
        font = pygame.font.Font(None, 64)
        msg = font.render("Press any key to start", True, black)
        self.screen.blit(msg, msg.get_rect(center=(screen_width//2, screen_height//2)))
        pygame.display.flip()

        waiting = True
        while waiting and not self.exit_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                    waiting = False
            time.sleep(0.01)

        # Učitaj pozadinu i resetuj sesiju
        self._load_background_image(self.bg_path)
        self._session_reset()

        clock = pygame.time.Clock()
        trail_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        trail_fade = 35
        trail_color = (*teal, 140)
        main_color = (*teal, 255)
        smoothed_pos = None

        # startuj data nit
        if not self.gazing_data_thread.is_alive():
            try:
                self.gazing_data_thread.start()
            except RuntimeError:
                pass

        running = True
        while running and not self.exit_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop_gazing()
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.stop_gazing()
                        running = False
                    elif event.key == pygame.K_q:
                        # završi i eksportuj
                        running = False
                    elif event.key == pygame.K_LEFTBRACKET:   # [
                        alpha = max(0.01, alpha - 0.05)
                        print(f"smoothing alpha -> {alpha:.2f}")
                    elif event.key == pygame.K_RIGHTBRACKET:  # ]
                        alpha = min(0.99, alpha + 0.05)
                        print(f"smoothing alpha -> {alpha:.2f}")
                    elif event.key == pygame.K_MINUS:         # -
                        trail_fade = max(0, trail_fade - 5)
                        print(f"trail fade -> {trail_fade}")
                    elif event.key == pygame.K_EQUALS:        # =
                        trail_fade = min(255, trail_fade + 5)
                        print(f"trail fade -> {trail_fade}")

            if self.bg_surface is not None:
                self.screen.blit(self.bg_surface, (0, 0))
            else:
                self.screen.fill((240, 240, 240))

            trail_surface.fill((0, 0, 0, trail_fade), special_flags=pygame.BLEND_RGBA_SUB)

            # prikaz tačke i zapis u bafer
            if self.gaze_tracker.gaze is not None:
                x, y = self.gaze_tracker.gaze
                if smoothed_pos is None:
                    smoothed_pos = [x, y]
                else:
                    smoothed_pos[0] = alpha * x + (1 - alpha) * smoothed_pos[0]
                    smoothed_pos[1] = alpha * y + (1 - alpha) * smoothed_pos[1]

                ix, iy = int(smoothed_pos[0]), int(smoothed_pos[1])

                pygame.draw.circle(trail_surface, trail_color, (ix, iy), dot_radius)
                self.screen.blit(trail_surface, (0, 0))
                pygame.draw.circle(self.screen, main_color, (ix, iy), dot_radius)

                # snimi uzorak (float pre int je OK)
                self._record_gaze_point(smoothed_pos[0], smoothed_pos[1])

            pygame.display.flip()
            clock.tick(60)

        # Nakon izlaska iz petlje – eksport (PNG + CSV) i graceful shutdown
        try:
            # method_label prema method_num
            label = 'Linearno_mapiranje' if method_num == 0 else (
                    'Polinomijalna_regresija' if method_num == 1 else 'SVR')
            self._finalize_and_save(method_label=label)
        except Exception as e:
            print(f"[Gazing] Greška pri eksportu: {e}")

        pygame.quit()
        print('Exiting Gazing (GUI)')

    # ---------- DATA THREAD (predict) ----------

    def get_and_estimate_data(self):
        from queue import Empty
        while not self.exit_event.is_set():
            t0 = time.perf_counter()
            try:
                gaze = None  # inicijalizacija na početku

                lm = None
                try:
                    lm = self.gaze_tracker.cam.eyes_landmarks
                except Exception:
                    lm = None

                eye_center_input = None
                eye_outer_input = None
                if lm:
                    eye_center_input = lm.get("l_iris_center")
                    eye_outer_input = lm.get("left_eye")

                l_corner_input = r_corner_input = None
                try:
                    l_corner_input, r_corner_input = self.gaze_tracker.detector.get_eye_corners(eye_outer_input)
                except Exception:
                    l_corner_input, r_corner_input = None, None

                method = method_num  # 0=linear,1=poly,2=svr
                if method == 0:
                    if eye_center_input is None:
                        time.sleep(0.001)
                        continue
                    gaze = self.gaze_tracker.linear_mapping(eye_center_input)
                elif method == 1:
                    if (eye_center_input is None) or (l_corner_input is None) or (r_corner_input is None):
                        time.sleep(0.001)
                        continue
                    gaze = self.gaze_tracker.polynomial_mapping(eye_center_input, l_corner_input, r_corner_input)
                else:
                    if (eye_center_input is None) or (l_corner_input is None) or (r_corner_input is None):
                        time.sleep(0.001)
                        continue
                    gaze = self.gaze_tracker.svr_mapping(eye_center_input, l_corner_input, r_corner_input)

                if gaze is not None:
                    self.gaze_tracker.gaze = (float(gaze[0]), float(gaze[1]))

            except Exception:
                import traceback
                print("[Gazing.get_and_estimate_data] exception:\n", traceback.format_exc())
                time.sleep(0.01)

            dt = time.perf_counter() - t0
            if dt < 0.001:
                time.sleep(0.001)

    def draw_gaze(self):
        while not self.exit_event.is_set():
            time.sleep(0.05)

    def stop_gazing(self):
        self.exit_event.set()
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
        print('Exiting Gazing')
