import cv2
import mediapipe as mp
import pygame
import numpy as np
import math
import random
import time
import urllib.request
from pathlib import Path

# ============================================================
# HAND PARTICLE VISUALIZER
# Compatible with modern MediaPipe Tasks API + Python 3.13
# ============================================================

CAM_W, CAM_H = 640, 480
PANEL_W, PANEL_H = 640, 480
WINDOW_W, WINDOW_H = CAM_W + PANEL_W, CAM_H
N_PARTICLES = 1800

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100000:
        return

    print("Model tangan belum ada.")
    print("Sedang download hand_landmarker.task...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model berhasil didownload.")
    except Exception as e:
        print("\nGagal download model otomatis.")
        print("Download file hand_landmarker.task dari:")
        print(MODEL_URL)
        print("Simpan file tersebut di folder yang sama dengan hand_particles.py")
        raise RuntimeError("Model MediaPipe tidak tersedia.") from e


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def classify_hand(lm):
    """
    lm = 21 normalized landmarks dari MediaPipe.
    Return:
      HEART  = ibu jari + telunjuk berdekatan
      SCATTER = mengepal
      VORTEX = telapak terbuka
      IDLE = lainnya
    """

    # Ujung jari
    thumb = (lm[4].x, lm[4].y)
    index = (lm[8].x, lm[8].y)

    # Jarak pinch relatif terhadap ukuran telapak
    palm = dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y))
    pinch = dist(thumb, index)

    if palm > 0 and pinch / palm < 0.55:
        return "HEART"

    # Cek jari terbuka menggunakan jarak tip dari wrist.
    wrist = np.array([lm[0].x, lm[0].y])
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    extended = 0
    for tip, pip in zip(tips, pips):
        tip_d = np.linalg.norm(np.array([lm[tip].x, lm[tip].y]) - wrist)
        pip_d = np.linalg.norm(np.array([lm[pip].x, lm[pip].y]) - wrist)
        if tip_d > pip_d * 1.12:
            extended += 1

    if extended <= 1:
        return "SCATTER"

    if extended >= 4:
        return "VORTEX"

    return "IDLE"


class Particle:
    def __init__(self, w, h):
        self.base = np.array([
            random.uniform(0, w),
            random.uniform(0, h)
        ], dtype=float)
        self.pos = self.base.copy()
        self.vel = np.zeros(2, dtype=float)
        self.phase = random.random() * math.tau
        self.size = random.choice([1, 1, 1, 2, 2, 3])

    def update(self, mode, target, t, w, h):
        p = self.pos
        v = self.vel

        if mode == "VORTEX":
            cx, cy = target
            dx = p[0] - cx
            dy = p[1] - cy
            r = max(20.0, math.hypot(dx, dy))

            # Tarik ke pusat + putar mengelilingi pusat
            v[0] += (-dx / r) * 0.16 + (-dy / r) * 0.12
            v[1] += (-dy / r) * 0.16 + ( dx / r) * 0.12

        elif mode == "HEART":
            # Bentuk heart parametric di sekitar target.
            u = (self.phase + t * 0.25) % math.tau
            hx = 16 * math.sin(u) ** 3
            hy = -(13 * math.cos(u) - 5 * math.cos(2*u)
                   - 2 * math.cos(3*u) - math.cos(4*u))

            scale = min(w, h) / 38.0
            goal = np.array([
                target[0] + hx * scale,
                target[1] + hy * scale
            ])

            v += (goal - p) * 0.018

        elif mode == "SCATTER":
            cx, cy = target
            dx = p[0] - cx
            dy = p[1] - cy
            r = max(8.0, math.hypot(dx, dy))

            # Dorong partikel menjauh dari tangan.
            v[0] += (dx / r) * 1.0
            v[1] += (dy / r) * 1.0

        else:
            # Kembali perlahan ke posisi awal.
            v += (self.base - p) * 0.0025

        v *= 0.93
        p += v

        # Batas layar
        if p[0] < 0:
            p[0] = 0
            v[0] *= -0.6
        elif p[0] >= w:
            p[0] = w - 1
            v[0] *= -0.6

        if p[1] < 0:
            p[1] = 0
            v[1] *= -0.6
        elif p[1] >= h:
            p[1] = h - 1
            v[1] *= -0.6


def main():
    ensure_model()

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Hand Particle Visualizer - MediaPipe Tasks")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22, bold=True)

    # Kamera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        pygame.quit()
        raise RuntimeError(
            "Kamera tidak bisa dibuka. Pastikan izin kamera Windows aktif."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    particles = [Particle(PANEL_W, PANEL_H) for _ in range(N_PARTICLES)]

    mode = "IDLE"
    target = (PANEL_W // 2, PANEL_H // 2)
    start_time = time.monotonic()
    frame_id = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            running = True

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            running = False

                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    mode = classify_hand(lm)

                    # Posisi telunjuk -> koordinat panel.
                    x = int(lm[8].x * PANEL_W)
                    y = int(lm[8].y * PANEL_H)
                    target = (
                        max(0, min(PANEL_W - 1, x)),
                        max(0, min(PANEL_H - 1, y))
                    )

                    # Gambar titik landmark sederhana di kamera.
                    for p in lm:
                        px = int(p.x * CAM_W)
                        py = int(p.y * CAM_H)
                        cv2.circle(frame, (px, py), 3, (80, 220, 255), -1)
                else:
                    mode = "IDLE"

                # Update partikel
                t = time.monotonic() - start_time
                for particle in particles:
                    particle.update(mode, target, t, PANEL_W, PANEL_H)

                # ---------- CAMERA PANEL ----------
                camera_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_surface = pygame.surfarray.make_surface(
                    np.transpose(camera_rgb, (1, 0, 2))
                )

                screen.blit(camera_surface, (0, 0))

                # ---------- PARTICLE PANEL ----------
                panel = pygame.Surface((PANEL_W, PANEL_H))
                panel.fill((3, 5, 12))

                # Glow titik dengan beberapa layer ringan.
                for particle in particles:
                    x = int(particle.pos[0])
                    y = int(particle.pos[1])
                    if 0 <= x < PANEL_W and 0 <= y < PANEL_H:
                        pygame.draw.circle(
                            panel,
                            (90, 190, 255),
                            (x, y),
                            particle.size
                        )

                # Target tangan
                if mode != "IDLE":
                    pygame.draw.circle(
                        panel,
                        (255, 255, 255),
                        target,
                        5,
                        1
                    )

                screen.blit(panel, (CAM_W, 0))

                # Divider
                pygame.draw.line(
                    screen,
                    (45, 45, 55),
                    (CAM_W, 0),
                    (CAM_W, WINDOW_H),
                    2
                )

                # Label
                label = font.render(
                    f"GESTURE: {mode}",
                    True,
                    (235, 235, 235)
                )
                screen.blit(label, (CAM_W + 18, 15))

                help_text = font.render(
                    "Q / ESC = keluar",
                    True,
                    (180, 180, 180)
                )
                screen.blit(help_text, (CAM_W + 18, WINDOW_H - 38))

                pygame.display.flip()
                clock.tick(60)
                frame_id += 1

    finally:
        cap.release()
        pygame.quit()


if __name__ == "__main__":
    main()
