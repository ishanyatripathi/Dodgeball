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
GRAY = (200, 200, 200)
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
sound_dodged = pygame.mixer.Sound("dodged.mp3")
sound_hit = pygame.mixer.Sound("hit.mp3")
sound_gameover = pygame.mixer.Sound("gameover.mp3")

# === Initialize MediaPipe ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75)
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
            self.x = random.randint(PROJECTILE_RADIUS, SCREEN_WIDTH - PROJECTILE_RADIUS)
            self.y = 0
            self.speed_x = 0
            self.speed_y = random.randint(5, 10)
        else:
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

def get_landmark_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def game_over_screen(score):
    sound_gameover.play()
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

        # Default positions
        green_pos = green_tracker.positions[-1] if green_tracker.positions else (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        blue_pos = blue_tracker.positions[-1] if blue_tracker.positions else (SCREEN_WIDTH//2, SCREEN_HEIGHT//4)

        torso_box = None
        legs_box = None

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Green = hips center
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
            green_x = int(((left_hip.x + right_hip.x) / 2) * SCREEN_WIDTH)
            green_y = int(((left_hip.y + right_hip.y) / 2) * SCREEN_HEIGHT)
            green_pos = green_tracker.update((green_x, green_y))

            # Blue = nose
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            blue_x = int(nose.x * SCREEN_WIDTH)
            blue_y = int(nose.y * SCREEN_HEIGHT)
            blue_pos = blue_tracker.update((blue_x, blue_y))

            # Full-body rectangles
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

            s1 = get_landmark_coords(left_shoulder, SCREEN_WIDTH, SCREEN_HEIGHT)
            s2 = get_landmark_coords(right_shoulder, SCREEN_WIDTH, SCREEN_HEIGHT)
            h1 = get_landmark_coords(left_hip, SCREEN_WIDTH, SCREEN_HEIGHT)
            h2 = get_landmark_coords(right_hip, SCREEN_WIDTH, SCREEN_HEIGHT)
            a1 = get_landmark_coords(left_ankle, SCREEN_WIDTH, SCREEN_HEIGHT)
            a2 = get_landmark_coords(right_ankle, SCREEN_WIDTH, SCREEN_HEIGHT)

            # Torso rectangle
            torso_min_x = min(s1[0], s2[0], h1[0], h2[0])
            torso_max_x = max(s1[0], s2[0], h1[0], h2[0])
            torso_min_y = min(s1[1], s2[1])
            torso_max_y = max(h1[1], h2[1])
            torso_box = pygame.Rect(torso_min_x, torso_min_y, torso_max_x - torso_min_x, torso_max_y - torso_min_y)

            # Legs rectangle
            legs_min_x = min(h1[0], h2[0], a1[0], a2[0])
            legs_max_x = max(h1[0], h2[0], a1[0], a2[0])
            legs_min_y = min(h1[1], h2[1])
            legs_max_y = max(a1[1], a2[1])
            legs_box = pygame.Rect(legs_min_x, legs_min_y, legs_max_x - legs_min_x, legs_max_y - legs_min_y)

        # Move projectile
        projectile.move()

        # Collision detection
        def check_collision(px, py, cx, cy, radius):
            dx = px - cx
            dy = py - cy
            dist = (dx*dx + dy*dy)**0.5
            return dist < radius + PROJECTILE_RADIUS

        # Check bounding boxes
        box_collision = False
        if torso_box and torso_box.collidepoint(projectile.x, projectile.y):
            box_collision = True
        elif legs_box and legs_box.collidepoint(projectile.x, projectile.y):
            box_collision = True

        # Fallback to head/hip
        circle_collision = (
            check_collision(projectile.x, projectile.y, green_pos[0], green_pos[1], PLAYER_RADIUS) or
            check_collision(projectile.x, projectile.y, blue_pos[0], blue_pos[1], PLAYER_RADIUS)
        )

        if box_collision or circle_collision:
            lives -= 1
            feedback_text = "HIT"
            feedback_color = RED
            feedback_timer = pygame.time.get_ticks()
            projectile.reset()
            projectile.from_top = not projectile.from_top
            sound_hit.play()

        elif projectile.is_offscreen():
            score += 1
            feedback_text = "DODGED"
            feedback_color = GREEN
            feedback_timer = pygame.time.get_ticks()
            projectile.reset()
            projectile.from_top = not projectile.from_top
            sound_dodged.play()

        # Draw everything
        screen.fill(BLACK)
        webcam_surface = cv_frame_to_pygame(frame)
        screen.blit(webcam_surface, (0, 0))

        pygame.draw.circle(screen, GREEN, green_pos, PLAYER_RADIUS)
        pygame.draw.circle(screen, BLUE, blue_pos, PLAYER_RADIUS)
        projectile.draw()

        if torso_box:
            pygame.draw.rect(screen, WHITE, torso_box, 2)
        if legs_box:
            pygame.draw.rect(screen, GRAY, legs_box, 2)

        score_text = font.render(f"Score: {score}", True, WHITE)
        lives_text = font.render(f"Lives: {lives}", True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (SCREEN_WIDTH - 120, 10))

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
