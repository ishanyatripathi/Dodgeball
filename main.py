import cv2
import mediapipe as mp
import pygame
import random
import time
import sys

# === Game Settings ===
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
PROJECTILE_RADIUS = 20
PLAYER_RADIUS = 30
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# === Initialize Pygame ===
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dodge the Projectile")
clock = pygame.time.Clock()
font = pygame.font.SysFont('timesnewroman', 32)
font.set_bold(True)

# === Initialize Mixer ===
pygame.mixer.init()

# Load sounds
sound_dodged = pygame.mixer.Sound("dodged.mp3")
sound_hit = pygame.mixer.Sound("hit.mp3")
sound_gameover = pygame.mixer.Sound("gameover.mp3")

# === Initialize MediaPipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)
mp_drawing = mp.solutions.drawing_utils

# === Tracker with smoothing ===
class Tracker:
    def __init__(self, window_size=5):
        self.positions = []
        self.window_size = window_size

    def update(self, new_pos):
        self.positions.append(new_pos)
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        avg_x = sum(p[0] for p in self.positions) / len(self.positions)
        avg_y = sum(p[1] for p in self.positions) / len(self.positions)
        return int(avg_x), int(avg_y)

# === Classes ===
class Projectile:
    def __init__(self, from_top=True):
        self.from_top = from_top
        self.reset()

    def reset(self):
        if self.from_top:
            # Spawn at random x at top, move downward
            self.x = random.randint(PROJECTILE_RADIUS, SCREEN_WIDTH - PROJECTILE_RADIUS)
            self.y = 0
            self.speed_x = 0
            self.speed_y = random.randint(5, 10)
        else:
            # Spawn at left side, random y, move rightward
            self.x = 0
            self.y = random.randint(PROJECTILE_RADIUS, SCREEN_HEIGHT // 2)
            self.speed_x = random.randint(5, 10)
            self.speed_y = 0

    def move(self):
        self.x += self.speed_x
        self.y += self.speed_y

    def draw(self):
        pygame.draw.circle(screen, RED, (self.x, self.y), PROJECTILE_RADIUS)

    def is_offscreen(self):
        return self.x > SCREEN_WIDTH or self.y > SCREEN_HEIGHT

# === Utility Functions ===
def cv_frame_to_pygame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
    frame = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    return frame

def game_over_screen(score):
    sound_gameover.play()  # play game over sound
    screen.fill(BLACK)
    msg1 = font.render("GAME OVER", True, RED)
    msg2 = font.render(f"Final Score: {score}", True, WHITE)
    msg3 = font.render("Press R to Restart or Q to Quit", True, WHITE)
    screen.blit(msg1, (SCREEN_WIDTH // 2 - msg1.get_width() // 2, SCREEN_HEIGHT // 3))
    screen.blit(msg2, (SCREEN_WIDTH // 2 - msg2.get_width() // 2, SCREEN_HEIGHT // 2))
    screen.blit(msg3, (SCREEN_WIDTH // 2 - msg3.get_width() // 2, SCREEN_HEIGHT // 1.5))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    main()
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()

# === Main Game ===
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    green_tracker = Tracker()
    blue_tracker = Tracker()

    score = 0
    lives = 3
    feedback_timer = 0
    feedback_text = ""
    feedback_color = WHITE

    # Projectile: start either from top or left randomly, but only one at a time
    projectile = Projectile(from_top=random.choice([True, False]))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to access webcam.")
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Green tracker = hips average (below chest)
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            green_x = int(((left_hip.x + right_hip.x) / 2) * SCREEN_WIDTH)
            green_y = int(((left_hip.y + right_hip.y) / 2) * SCREEN_HEIGHT)
            green_pos = green_tracker.update((green_x, green_y))

            # Blue tracker = nose (head)
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            blue_x = int(nose.x * SCREEN_WIDTH)
            blue_y = int(nose.y * SCREEN_HEIGHT)
            blue_pos = blue_tracker.update((blue_x, blue_y))
        else:
            # If no detection, keep previous positions
            green_pos = green_tracker.positions[-1] if green_tracker.positions else (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
            blue_pos = blue_tracker.positions[-1] if blue_tracker.positions else (SCREEN_WIDTH//2, SCREEN_HEIGHT//4)

        # Move projectile
        projectile.move()

        # Collision detection with both trackers (green and blue)
        def check_collision(px, py, cx, cy, radius):
            dx = px - cx
            dy = py - cy
            dist = (dx*dx + dy*dy)**0.5
            return dist < radius + PROJECTILE_RADIUS

        collision = (check_collision(projectile.x, projectile.y, green_pos[0], green_pos[1], PLAYER_RADIUS) or
                     check_collision(projectile.x, projectile.y, blue_pos[0], blue_pos[1], PLAYER_RADIUS))

        if collision:
            lives -= 1
            feedback_text = "HIT"
            feedback_color = RED
            feedback_timer = pygame.time.get_ticks()
            projectile.reset()
            # Alternate projectile direction
            projectile.from_top = not projectile.from_top
            sound_hit.play()  # play hit sound

        elif projectile.is_offscreen():
            score += 1
            feedback_text = "DODGED"
            feedback_color = GREEN
            feedback_timer = pygame.time.get_ticks()
            projectile.reset()
            projectile.from_top = not projectile.from_top
            sound_dodged.play()  # play dodged sound

        # Draw everything
        screen.fill(BLACK)
        webcam_surface = cv_frame_to_pygame(frame)
        screen.blit(webcam_surface, (0, 0))

        # Draw trackers
        pygame.draw.circle(screen, GREEN, green_pos, PLAYER_RADIUS)
        pygame.draw.circle(screen, BLUE, blue_pos, PLAYER_RADIUS)

        # Draw projectile
        projectile.draw()

        # Draw UI
        score_text = font.render(f"Score: {score}", True, WHITE)
        lives_text = font.render(f"Lives: {lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (SCREEN_WIDTH - 120, 10))

        # Feedback text
        if pygame.time.get_ticks() - feedback_timer < 1000 and feedback_text:
            fb_surf = font.render(feedback_text, True, feedback_color)
            screen.blit(fb_surf, (SCREEN_WIDTH // 2 - fb_surf.get_width() // 2, 50))

        pygame.display.flip()
        clock.tick(FPS)

        if lives <= 0:
            cap.release()
            game_over_screen(score)

if __name__ == "__main__":
    main()
