import cv2
import mediapipe as mp
import pygame
import random
import sys

# === Game Settings ===
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
PROJECTILE_RADIUS = 20
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

# === Initialize Pygame ===
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dodge the Projectile - Skeleton Edition")
clock = pygame.time.Clock()
font = pygame.font.SysFont('timesnewroman', 28)
font.set_bold(True)

# === Initialize Mixer ===
pygame.mixer.init()
# Use placeholder sounds if files don't exist
try:
    sound_dodged = pygame.mixer.Sound("dodged.mp3")
    sound_hit = pygame.mixer.Sound("hit.mp3")
    sound_gameover = pygame.mixer.Sound("gameover.mp3")
except:
    # Create silent sounds if files are missing
    sound_dodged = pygame.mixer.Sound(buffer=bytearray([]))
    sound_hit = pygame.mixer.Sound(buffer=bytearray([]))
    sound_gameover = pygame.mixer.Sound(buffer=bytearray([]))

# === Initialize MediaPipe Holistic ===
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Custom drawing specs for better visibility
landmark_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
connection_drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True
)

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
            self.speed_y = random.randint(5, 9)
        else:
            self.x = 0
            self.y = random.randint(PROJECTILE_RADIUS, SCREEN_HEIGHT // 2)
            self.speed_x = random.randint(5, 9)
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
    return pygame.surfarray.make_surface(frame.swapaxes(0, 1))

def get_landmark_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def get_skeleton_connections(results, width, height):
    """Get all skeleton connections from MediaPipe Holistic"""
    connections = []
    
    # Pose connections
    if results.pose_landmarks:
        pose_connections = mp_holistic.POSE_CONNECTIONS
        for connection in pose_connections:
            start_idx, end_idx = connection
            if (start_idx < len(results.pose_landmarks.landmark) and 
                end_idx < len(results.pose_landmarks.landmark)):
                start_point = get_landmark_coords(results.pose_landmarks.landmark[start_idx], width, height)
                end_point = get_landmark_coords(results.pose_landmarks.landmark[end_idx], width, height)
                connections.append((start_point, end_point))
    
    # Left hand connections
    if results.left_hand_landmarks:
        hand_connections = mp_holistic.HAND_CONNECTIONS
        for connection in hand_connections:
            start_idx, end_idx = connection
            if (start_idx < len(results.left_hand_landmarks.landmark) and 
                end_idx < len(results.left_hand_landmarks.landmark)):
                start_point = get_landmark_coords(results.left_hand_landmarks.landmark[start_idx], width, height)
                end_point = get_landmark_coords(results.left_hand_landmarks.landmark[end_idx], width, height)
                connections.append((start_point, end_point))
    
    # Right hand connections
    if results.right_hand_landmarks:
        hand_connections = mp_holistic.HAND_CONNECTIONS
        for connection in hand_connections:
            start_idx, end_idx = connection
            if (start_idx < len(results.right_hand_landmarks.landmark) and 
                end_idx < len(results.right_hand_landmarks.landmark)):
                start_point = get_landmark_coords(results.right_hand_landmarks.landmark[start_idx], width, height)
                end_point = get_landmark_coords(results.right_hand_landmarks.landmark[end_idx], width, height)
                connections.append((start_point, end_point))
    
    # Face connections (simplified - use face contours)
    if results.face_landmarks:
        face_connections = mp_holistic.FACEMESH_CONTOURS
        # Only use every 3rd connection to reduce complexity
        for i, connection in enumerate(face_connections):
            if i % 3 == 0:  # Sample every 3rd connection
                start_idx, end_idx = connection
                if (start_idx < len(results.face_landmarks.landmark) and 
                    end_idx < len(results.face_landmarks.landmark)):
                    start_point = get_landmark_coords(results.face_landmarks.landmark[start_idx], width, height)
                    end_point = get_landmark_coords(results.face_landmarks.landmark[end_idx], width, height)
                    connections.append((start_point, end_point))
    
    return connections

def check_collision_with_skeleton(projectile, skeleton_connections):
    """Check if projectile collides with any skeleton bone"""
    for connection in skeleton_connections:
        line_start, line_end = connection
        
        # Calculate distance from projectile to line segment
        line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
        line_len_sq = line_vec[0]**2 + line_vec[1]**2
        
        if line_len_sq == 0:
            # Line is actually a point
            dx = projectile.x - line_start[0]
            dy = projectile.y - line_start[1]
            if (dx*dx + dy*dy) <= (PROJECTILE_RADIUS ** 2):
                return True
            continue
        
        # Calculate projection of projectile onto the line
        t = max(0, min(1, ((projectile.x - line_start[0]) * line_vec[0] + 
                           (projectile.y - line_start[1]) * line_vec[1]) / line_len_sq))
        
        # Find the closest point on the line segment
        closest_x = line_start[0] + t * line_vec[0]
        closest_y = line_start[1] + t * line_vec[1]
        
        # Calculate distance between projectile and closest point
        dx = projectile.x - closest_x
        dy = projectile.y - closest_y
        dist_sq = dx*dx + dy*dy
        
        if dist_sq <= (PROJECTILE_RADIUS ** 2):
            return True
    
    return False

def draw_skeleton_pygame(skeleton_connections):
    """Draw the skeleton using pygame (on top of webcam feed)"""
    for connection in skeleton_connections:
        line_start, line_end = connection
        pygame.draw.line(screen, YELLOW, line_start, line_end, 3)
        
        # Draw small circles at joints for better visibility
        pygame.draw.circle(screen, GREEN, line_start, 4)
        pygame.draw.circle(screen, GREEN, line_end, 4)

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
        results = holistic.process(rgb_frame)

        # Get skeleton connections
        skeleton_connections = get_skeleton_connections(results, SCREEN_WIDTH, SCREEN_HEIGHT)

        # Draw MediaPipe landmarks on OpenCV frame (optional, for background)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=landmark_drawing_spec,
                connection_drawing_spec=connection_drawing_spec
            )

        # Move projectile
        projectile.move()

        # Collision detection with skeleton
        hit = check_collision_with_skeleton(projectile, skeleton_connections)

        if hit:
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

        # Draw skeleton with pygame (on top of webcam feed)
        draw_skeleton_pygame(skeleton_connections)

        # Draw projectile
        projectile.draw()

        # Draw UI
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
