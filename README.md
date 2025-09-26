Dodge the Projectile Game

**Dodge the Projectile** is an interactive game where the player dodges falling projectiles using head or hip movements, tracked by a webcam. 

**Requirements:**
<br>Python 3.x <br/>
Libraries: opencv-python, mediapipe, pygame

<br>Install the required libraries with:<br/>
pip install opencv-python mediapipe pygame

**<br>How to Run:<br/>**
Clone or download the project and ensure the sound files dodged.mp3, hit.mp3, and gameover.mp3 are in the same directory as the script.

**<br>Run the game:<br/>**
python main.py


Use your webcam to control the player’s movement by moving your head (nose) or hips (average of left and right hip positions).
Press 'R' to restart or 'Q' to quit after a game over.

**<br>MediaPipe Landmarks Used<br/>**
The game tracks two key body parts using MediaPipe Pose landmarks:

**<br>Green Tracker (hips):<br/>**
Average of LEFT_HIP and RIGHT_HIP

**<br>Blue Tracker (head):<br/>**
NOSE

<br> **Watch Demo**<br/>
[Gameplay Video](https://drive.google.com/file/d/1JFrgUcLlCMQeUTNEicPZJHKBQ7cMgn_4/view?usp=sharing)

**<br/>Author: <br/>**
Ishanya Triapthi
